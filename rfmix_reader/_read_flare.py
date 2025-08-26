"""
Adapted from `_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py
"""
import warnings
from re import search
from tqdm import tqdm
from glob import glob
from numpy import int32
from dask.array import Array, concatenate
from pysam import tabix_index, VariantFile
from collections import OrderedDict as odict
from os.path import basename, dirname, join, exists, getsize
from typing import Optional, Callable, List, Tuple, Dict, Iterator

from ._chunk import Chunk
from ._fb_read import read_fb
from ._utils import get_prefixes
from ._utils import set_gpu_environment
from ._errorhandling import BinaryFileNotFoundError

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, read_csv, concat, CategoricalDtype
else:
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
    loci : :class:`pandas.DataFrame`
        Loci information for the FB data.
    g_anc : :class:`pandas.DataFrame`
        Global ancestry by chromosome from Flare. This is the same as previous
        name, `rf_q`.
    admix : :class:`dask.array.Array`
        Local ancestry per population (columns pop1*nsamples ... popX*nsamples).
        This is in order of the populations see `g_anc`.

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
    pbar = tqdm(desc="Mapping loci info files", total=len(fn),
                disable=not verbose)
    loci = _read_file(fn, lambda f: _read_loci(f["anc.vcf.gz"], chunk_size),
                      pbar)
    pbar.close()

    # Adjust loci indices and concatenate
    nmarkers = {}; index_offset = 0
    for i, bi in enumerate(loci):
        nmarkers[fn[i]["anc.vcf.gz"]] = bi.shape[0]
        bi["i"] += index_offset
        index_offset += bi.shape[0]
    loci = concat(loci, axis=0, ignore_index=True)

    # Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn), disable=not verbose)
    g_anc = _read_file(fn, lambda f: _read_anc(f["global.anc.gz"]), pbar)
    pbar.close()

    nsamples = g_anc[0].shape[0]
    pops = g_anc[0].drop(["sample_id", "chrom"], axis=1).columns.values
    g_anc = concat(g_anc, axis=0, ignore_index=True)

    # Loading local ancestry by loci
    if generate_binary:
        create_binaries(file_prefix, binary_dir)

    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn), disable=not verbose)
    admix = _read_file(
        fn,
        lambda f: _read_fb(f["anc.vcf.gz"], nsamples,
                           nmarkers[f["anc.vcf.gz"]], pops,
                           binary_dir, Chunk()),
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
       chunks = list(_load_vcf_data(fn, chunk_size))
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


def _read_fb(fn: str, nsamples: int, nloci: int, pops: list,
             temp_dir: str, chunk: Optional[Chunk] = None) -> Array:
    """
    Read the forward-backward matrix from a file as a Dask Array.

    Parameters:
    ----------
    fn (str): The file path of the forward-backward matrix file.
    nsamples (int): The number of samples in the dataset.
    nloci (int): The number of loci in the dataset.
    pops (list): A list of population labels.
    chunk (Chunk, optional): A Chunk object specifying the chunk size for reading.

    Returns:
    -------
    dask.array.Array: The forward-backward matrix as a Dask Array.
    """
    npops = len(pops)
    nrows = nloci
    ncols = (nsamples * npops * 2)
    row_chunk = nrows if chunk.nloci is None else min(nrows, chunk.nloci)
    col_chunk = ncols if chunk.nsamples is None else min(ncols, chunk.nsamples)
    max_npartitions = 16_384
    row_chunk = max(nrows // max_npartitions, row_chunk)
    col_chunk = max(ncols // max_npartitions, col_chunk)
    binary_fn = join(temp_dir,
                     basename(fn).split(".")[0] + ".bin")
    if exists(binary_fn):
        X = read_fb(binary_fn, nrows, ncols, row_chunk, col_chunk)
    else:
        raise BinaryFileNotFoundError(binary_fn, temp_dir)
    # Subset populations and sum adjacent columns
    return _subset_populations(X, npops)


def _subset_populations(X: Array, npops: int) -> Array:
    """
    Subset and process the input array X based on populations.

    Parameters:
    X (dask.array): Input array where columns represent data for different populations.
    npops (int): Number of populations for column processing.

    Returns:
    dask.array: Processed array with adjacent columns summed for each population subset.
    """
    pop_subset = []
    pop_start = 0
    ncols = X.shape[1]
    if ncols % npops != 0:
        raise ValueError("The number of columns in X must be divisible by npops.")
    while pop_start < npops:
        X0 = X[:, pop_start::npops] # Subset based on populations
        if X0.shape[1] % 2 != 0:
            raise ValueError("Number of columns must be even.")
        X0_summed = X0[:, ::2] + X0[:, 1::2] # Sum adjacent columns
        pop_subset.append(X0_summed)
        pop_start += 1
    return concatenate(pop_subset, 1, True)


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


def _load_vcf_data(vcf_file: str, chunk_size: int32 = 1_000_000
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
