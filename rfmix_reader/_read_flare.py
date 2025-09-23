"""
Revision of `_read_rfmix.py` to work with FLARE output data.
"""
from re import search
from tqdm import tqdm
from cyvcf2 import VCF
from numpy import int32
from dask import delayed
from os.path import exists
from re import match as rmatch
from typing import List, Tuple, Iterator
from collections import OrderedDict as odict
from dask.array import Array, concatenate, from_delayed, stack

from ._utils import get_prefixes, set_gpu_environment, _read_file

try:
    from torch.cuda import is_available as gpu_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def gpu_available():
        return False

if gpu_available():
    from cupy import array, full, zeros, asarray, int8
    from cudf import DataFrame, read_csv, concat, CategoricalDtype
else:
    from numpy import array, full, zeros, asarray, int8
    from pandas import DataFrame, read_csv, concat, CategoricalDtype

__all__ = ["read_flare"]

def read_flare(
        file_prefix: str, chunk_size: int32 = 1_000_000, verbose: bool = True,
) -> Tuple[DataFrame, DataFrame, Array]:
    """
    Read Flare files into data frames and a Dask array.

    Parameters
    ----------
    file_prefix : str
        Path prefix to the set of Flare files. It will load all of the chromosomes
        at once.
    chunk_size : int
        Number of records to read per chunk.
    verbose : bool, optional
        :const:`True` for progress information; :const:`False` otherwise.
        Default:`True`.

    Returns
    -------
    loci_df : :class:`DataFrame`
        Loci information for the FB data.
    g_anc : :class:`DataFrame`
        Global ancestry by chromosome from Flare.
    local_array : :class:`dask.array.Array`
        Local ancestry per population stacked (variants, samples, ancestries).
        This is in alphabetical order of the populations. This matches RFMix.

    Notes
    -----
    Local ancestry output will be either :const:`0`, :const:`1`, :const:`2`, or
    :data:`math.nan`:

    - :const:`0` No alleles are associated with this ancestry
    - :const:`1` One allele is associated with this ancestry
    - :const:`2` Both alleles are associated with this ancestry
    """
    # Device information
    if verbose and gpu_available():
        set_gpu_environment()

    # Get file prefixes
    fn = get_prefixes(file_prefix, "flare", verbose)

    # Load loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn),
                disable=not verbose)
    loci_dfs = _read_file(fn, lambda f: _read_loci(f["anc.vcf.gz"], chunk_size),
                      pbar)
    pbar.close()

    index_offset = 0
    for df in loci_dfs: # Modify in-place
        df["i"] = range(index_offset, index_offset + df.shape[0])
        index_offset += df.shape[0]
    loci_df = concat(loci_dfs, axis=0, ignore_index=True)

    # Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn),
                disable=not verbose)
    g_anc = _read_file(fn, lambda f: _read_anc(f["global.anc.gz"]), pbar)
    pbar.close()
    g_anc = concat(g_anc, axis=0, ignore_index=True)

    # Loading local ancestry by loci
    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn),
                disable=not verbose)
    local_array = _read_file(
        fn,
        lambda f: _load_haplotypes(f["anc.vcf.gz"], int(chunk_size / 100)),
        pbar
    )
    pbar.close()
    local_array = concatenate(local_array, axis=0)
    return loci_df, g_anc, local_array


def _read_vcf(fn: str, chunk_size: int32 = 1_000_000) -> DataFrame:
    """
    Read a VCF file into a DataFrame.

    Parameters:
    ----------
    fn : str
        File name of the VCF file.
    chunk_size : int
        Number of records to include per chunk.

    Returns:
    -------
    DataFrame: DataFrame containing specified columns from the VCF file.
    """
    header = {"chromosome": CategoricalDtype(), "physical_position": int32}
    try:
       chunks = list(_load_vcf_info(fn, chunk_size))
       df = concat(chunks, ignore_index=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fn} not found.")
    except Exception as e:
        raise IOError(f"Error reading file {fn}: {e}")

    # Validate that resulting DataFrame is correct type
    if not isinstance(df, DataFrame):
        raise ValueError(f"Expected a DataFrame but got {type(df)} instead.")
    # Ensure DataFrame contains correct columns
    if not all(column in df.columns for column in list(header.keys())):
        raise ValueError(f"DataFrame does not contain expected columns: {columns}")
    return df


def _read_loci(fn: str, chunk_size: int32 = 1_000_000) -> DataFrame:
    """
    Read loci information from a TSV file and add a sequential index column.

    Parameters:
    ----------
    fn : str
        The file path of the TSV file containing loci information.
    chunk_size : int
        Number of records to include per chunk.

    Returns:
    -------
    DataFrame: A DataFrame containing the loci information with an
               additional 'i' column for indexing.
    """
    df = _read_vcf(fn, chunk_size)
    df["i"] = range(df.shape[0])
    return df


def _read_csv(fn: str, header: dict) -> DataFrame:
    """
    Read a CSV file into a pandas DataFrame with specified data types.

    Parameters:
    ----------
    fn (str): The file path of the CSV file.
    header (dict): A dictionary mapping column names to data types.

    Returns:
    -------
    DataFrame: The data read from the CSV file as a pandas DataFrame.
    """
    try:
        df = read_csv(fn, sep="\t", names=list(header.keys()),
                      dtype=header, skiprows=1)
    except Exception as e:
        raise IOError(f"Error reading file '{fn}': {e}")

    # Validate that resulting DataFrame is correct type
    if not isinstance(df, DataFrame):
        raise ValueError(f"Expected a DataFrame but got {type(df)} instead.")
    return df


def _read_anc(fn: str) -> DataFrame:
    """
    Read the Q matrix from a file and add the chromosome information.

    Parameters:
    ----------
    fn (str): The file path of the Q matrix file.

    Returns:
    -------
    DataFrame: The Q matrix with the chromosome information added.
    """
    df = _read_anc_noi(fn)
    m = search(r'chr[\w]+', fn)
    if m:
        chrom = m.group(0)
        df["chrom"] = chrom
    else:
        print(f"Warning: Could not extract chromosome information from '{fn}'")
    return df


def _read_anc_noi(fn: str) -> DataFrame:
    """
    Read the Q matrix from a file without adding chromosome information.

    Parameters:
    ----------
    fn (str): The file path of the Q matrix file.

    Returns:
    -------
    DataFrame: The Q matrix without chromosome information.
    """
    try:
        header = odict(_types(fn))
        return _read_csv(fn, header)
    except Exception as e:
        raise IOError(f"Error reading Q matrix from '{fn}': {e}")


def _load_haplotypes(vcf_file: str, chunk_size: int32 = 10_000) -> Array:
    """
    Load haplotype ancestry counts from a VCF file into a stacked dask array.

    The function parses the `##ANCESTRY` header from the FLARE VCF to identify
    ancestries and their indices. It chunks variant records to efficiently
    process large files with minimal memory usage. For each variant and sample,
    it sums haplotype ancestries according to the following logic:
    - If both haplotype ancestries (AN1 and AN2) are identical, their count
      is summed and assigned to the single ancestry.
    - If different, each haplotype ancestry is counted individually.

    The ancestries are arranged alphabetically by their label. For example,
    with populations ["AFR", "EUR"], slice 0 corresponds to "AFR" and slice 1
    corresponds to "EUR".

    Parameters
    ----------
    vcf_file : str
        Path to the BGZF compressed and indexed VCF file containing haplotype
        ancestry information in FORMAT fields AN1 and AN2.

    chunk_size : int, optional
        Number of variant records to process per chunk for efficiency.
        Default is 10,000.

    Returns
    -------
    dask.array.Array
        A 3D stacked dask array with shape (num_variants, num_samples, num_ancestries),
        where the last dimension indexes ancestry populations alphabetically.

    Raises
    ------
    FileNotFoundError
        If the specified VCF file does not exist or cannot be opened.

    Examples
    --------
    >>> dask_array = _load_haplotypes("chr21.anc.vcf.gz", chunk_size=5_000)
    >>> print(dask_array.shape)
    (270000, 500, 2)
    >>> afr_counts = dask_array[:, :, 0]  # Access AFR ancestry slice
    """
    if not exists(vcf_file):
        raise FileNotFoundError(f"VCF file not found: {vcf_file}")

    vcf = VCF(vcf_file)
    samples = vcf.samples
    n_samples = len(samples)

    # Parse ancestry header
    ancestry_map = _parse_ancestry_header(vcf_file)
    n_ancestries = len(ancestry_map)

    def process_chunk(records):
        chunk_len = len(records)
        counts = zeros((chunk_len, n_samples, n_ancestries), dtype=int8)

        for i, rec in enumerate(records):
            an1 = rec.format("AN1")
            an2 = rec.format("AN2")

            # Mask missing values as -1
            an1 = asarray(an1, dtype=int8)
            an2 = asarray(an2, dtype=int8)

            # Count local ancestries in vectorized fashion
            for anc_idx in range(n_ancestries):
                counts[i, :, anc_idx] = (
                    (an1 == anc_idx).astype(int8) +
                    (an2 == anc_idx).astype(int8)
                )

        return counts

    records_buffer = []
    delayed_arrays = []
    for rec in vcf:
        records_buffer.append(rec)
        if len(records_buffer) == chunk_size:
            delayed_arrays.append(
                from_delayed(
                    delayed(process_chunk)(records_buffer),
                    shape=(chunk_size, n_samples, n_ancestries),
                    dtype=int8,
                )
            )
            records_buffer = []

    if records_buffer:
        delayed_arrays.append(
            from_delayed(
                delayed(process_chunk)(records_buffer),
                shape=(len(records_buffer), n_samples, n_ancestries),
                dtype=int8,
            )
        )

    # Build dask arrays by stacking (variants, samples, ancestries)
    combined = concatenate(delayed_arrays, axis=0)

    an_dask_arrays = {
        label: combined[:, :, ancestry_map[label]]
        for label in ancestry_map.keys()
    }
    arrays_list = [an_dask_arrays[k] for k in sorted(an_dask_arrays.keys())]

    return stack(arrays_list, axis=2)


def _parse_ancestry_header(vcf_file: str) -> dict:
    """
    Parse ancestry population index from the VCF header.

    Looks for a line starting with '##ANCESTRY=' formatted like:
    '##ANCESTRY=<EUR=0,AFR=1>'

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file.

    Returns
    -------
    dict
        Mapping from ancestry label (e.g., 'EUR') to integer index (e.g., 0).
    """
    vcf = VCF(vcf_file)
    ancestries = {}
    for hline in vcf.raw_header.splitlines():
        if hline.startswith("##ANCESTRY="):
            m = search(r"<(.+)>", hline)
            if m:
                pairs = m.group(1).split(",")
                for pair in pairs:
                    label, idx = pair.split("=")
                    ancestries[label] = int(idx)
            break
    return ancestries


def _load_vcf_info(vcf_file: str, chunk_size: int32 = 1_000_000
                   ) -> Iterator[DataFrame]:
    """
    Load VCF records from a BGZF compressed and convert to DataFrames.

    Parameters
    ----------
    vcf_file : str
        Path to BGZF compressed VCF (.vcf.gz) with an associated .tbi index.
    chunk_size : int
        Number of records to include per chunk.

    Yields
    ------
    DataFrame
        DataFrame with 'chromosome' and 'physical_position' columns loaded chunk.
    """
    vcf = VCF(vcf_file)
    records, count = [], 0

    for rec in vcf:
        records.append({'chromosome': rec.CHROM, 'physical_position': rec.POS})
        count += 1
        if count % chunk_size == 0:
            yield DataFrame(records)
            records = []

    if records:
        yield DataFrame(records)


def _types(fn: str) -> dict:
    """
    Infer the data types of columns in a TSV file.
    For FLARE global ancestry (global.anc.gz), force float32.

    Parameters:
    ----------
    fn (str) : File name of the TSV file.

    Returns:
    -------
    dict : Dictionary mapping column names to their inferred data types.
    """
    try:
        df = read_csv(fn, sep="\t", nrows=2)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{fn}' not found.")
    except Exception as e:
        raise IOError(f"Error reading file '{fn}': {e}")

    # Validate that the resulting DataFrame is of the correct type
    if not isinstance(df, DataFrame):
        raise ValueError(f"Expected a DataFrame but got {type(df)} instead.")
    # Ensure the DataFrame contains at least one column
    if df.shape[1] < 1:
        raise ValueError("The DataFrame does not contain any columns.")

    # Initialize the header dictionary with the sample_id column
    header = {"sample_id": CategoricalDtype()}
    # For global ancestry files, force float32
    if fn.endswith("global.anc.gz"):
        for col in df.columns[1:]:
            header[col] = 'float32'
    else:
        header.update(df.dtypes[1:].to_dict())

    return header
