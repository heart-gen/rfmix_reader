"""
Adapted from `_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py
"""
from re import search
from tqdm import tqdm
from numpy import int32
from dask import delayed
from os.path import exists
from re import match as rmatch
from pysam import tabix_index, VariantFile
from collections import OrderedDict as odict
from typing import Callable, List, Tuple, Iterator
from dask.array import Array, concatenate, from_delayed

from ._utils import get_prefixes, set_gpu_environment

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cupy import int32 as py_int32
    from cupy import array, full, zeros
    from cudf import DataFrame, read_csv, concat, CategoricalDtype
else:
    from numpy import int32 as py_int32
    from numpy import array, full, zeros
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
    loci : :class:`DataFrame`
        Loci information for the FB data.
    g_anc : :class:`DataFrame`
        Global ancestry by chromosome from Flare.
    admix : :class:`dask.array.Array`
        Local ancestry per population stacked (variants, samples, ancestries).
        This is in alphabetical order of the populations.

    Notes
    -----
    Local ancestry output will be either :const:`0`, :const:`1`, :const:`2`, or
    :data:`math.nan`:

    - :const:`0` No alleles are associated with this ancestry
    - :const:`1` One allele is associated with this ancestry
    - :const:`2` Both alleles are associated with this ancestry
    """
    # Device information
    if verbose and is_available():
        set_gpu_environment()

    # Get file prefixes
    fn = get_prefixes(file_prefix, "flare", verbose)

    # Load loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn),
                disable=not verbose)
    loci = _read_file(fn, lambda f: _read_loci(f["anc.vcf.gz"], chunk_size),
                      pbar)
    pbar.close()
    loci = concat(loci, axis=0, ignore_index=True)

    # Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn),
                disable=not verbose)
    g_anc = _read_file(fn, lambda f: _read_anc(f["global.anc.gz"]), pbar)
    pbar.close()
    g_anc = concat(g_anc, axis=0, ignore_index=True)

    # Loading local ancestry by loci
    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn),
                disable=not verbose)
    admix = _read_file(
        fn,
        lambda f: _load_haplotypes(f["anc.vcf.gz"], int(chunk_size / 100)),
        pbar,
    )
    pbar.close()
    admix = concatenate(admix, axis=0)
    return loci, g_anc, admix


def _read_file(fn: List[str], read_func: Callable, pbar=None) -> List:
    """
    Read data from multiple files using a provided read function.

    Parameters:
    ----------
    fn : List[str]
        A list of file paths to read.
    read_func : Callable
        A function to read data from each file.
    pbar : Optional
        A progress bar object to update during reading.

    Returns:
    -------
    List: A list containing the data read from each file.
    """
    data = [];
    for file_name in fn:
        data.append(read_func(file_name))
        if pbar:
            pbar.update(1)
    return data


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
    match = search(r'chr(\d+)', fn)
    if match:
        chrom = match.group(0)
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

    try:
        vcf = VariantFile(vcf_file)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error opening VCF file {vcf_file}: {e}")

    samples = list(vcf.header.samples)
    n_samples = len(samples)

    ancestries = _parse_ancestry_header(vcf_file)
    pop_labels = sorted(ancestries.keys())
    n_pop = len(pop_labels)
    idx_pop = {v: k for k, v in ancestries.items()}

    # Detect ancestry keys from first record
    first_record = next(vcf.fetch())
    an_keys = ['AN1', 'AN2'] # This is based on haplotypes
    vcf = VariantFile(vcf_file)  # reset iterator

    def process_chunk(records):
        chunk_len = len(records)
        # Create arrays; variants x samples x ancestries
        counts = zeros((chunk_len, n_samples, n_pop), dtype=py_int32)

        for i, record in enumerate(records):
            for j, sample in enumerate(samples):
                hap_val = [
                    record.samples[sample].get(an_key, -1) for an_key in an_keys
                ]

                # Sum counts for ancestry
                if all(h == hap_val[0] for h in hap_val):
                    anc_idx = hap_val[0]
                    if anc_idx in idx_pop:
                        counts[i, j, anc_idx] += len(hap_val)
                    else:
                        for anc_idx in hap_val:
                            if anc_idx in idx_pop:
                                counts[i, j, anc_idx] += 1
        return counts

    records_buffer = []
    delayed_arrays = []

    fetch_iter = vcf.fetch()
    for record in fetch_iter:
        records_buffer.append(record)
        if len(records_buffer) == chunk_size:
            delayed_arrays.append(delayed(process_chunk)(records_buffer))
            records_buffer = []

    if records_buffer:
        delayed_arrays.append(delayed(process_chunk)(records_buffer))

    # Build dask arrays by stacking
    combined = concatenate(
        [from_delayed(d, shape=(chunk_size, n_samples, n_pop),
                      dtype=py_int32) for d in delayed_arrays],
        axis=0,
    )

    an_dask_arrays = {
        label: combined[:, :, ancestries[label]] for label in pop_labels
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
    import gzip
    ancestries = {}
    with gzip.open(vcf_file, "rt") as f:
        for line in f:
            if line.startswith("##ANCESTRY="):
                m = search(r"<(.+)>", line)
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
    Load VCF records from a BGZF compressed and tabix indexed VCF file in chunks
    using pysam and convert to DataFrames.

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
    _create_tabix(vcf_file)
    vcf = VariantFile(vcf_file)
    records = []; count = 0
    fetch_iter = vcf.fetch()

    for rec in fetch_iter:
        records.append({'chromosome': rec.chrom, 'physical_position': rec.pos})
        count += 1
        if count % chunk_size == 0:
            yield DataFrame(records)
            records = []

    if records:
        yield DataFrame(records)


def _create_tabix(vcf_file: str) -> None:
    """
    Checks for the presence of a .tbi file for the given VCF.
    If missing, create a tabix index using pysam.tabix_index.

    Parameters
    ----------
    vcf_file : str
        Path to the bgzip-compressed VCF file (.vcf.gz).

    Raises
    ------
    RuntimeError
        If indexing fails or input file is not properly bgzip compressed or sorted.
    """
    tbi_path = vcf_file + ".tbi"
    if not exists(tbi_path):
        print(f"Index file {tbi_path} not found, creating tabix index...")
        try:
            tabix_index(vcf_file, preset="vcf", force=True)
            print("Tabix index created successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create tabix index for {vcf_file}: {str(e)}"
            )


def _types(fn: str) -> dict:
    """
    Infer the data types of columns in a TSV file.

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
    header.update(df.dtypes[1:].to_dict())
    return header
