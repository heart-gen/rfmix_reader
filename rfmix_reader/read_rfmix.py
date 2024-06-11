"""
Adapted from `_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py
"""
import warnings
from glob import glob
from os.path import basename, dirname, join
from collections import OrderedDict as odict
from typing import Optional, Callable, List, Tuple

from dask.array import Array

from .chunk import Chunk
from .fb_read import read_fb

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    from cudf import DataFrame, read_csv, concat
    set_gpu_environment()
else:
    from pandas import DataFrame, read_csv, concat

__all__ = ["read_rfmix"]

def read_rfmix(
        file_prefix: str, verbose: bool = True
) -> Tuple[DataFrame, DataFrame, Array]:
    """
    Read RFMix files into data frames and a Dask array.

    Notes
    -----
    Local ancestry can be either :const:`0`, :const:`1`, :const:`2`, or
    :data:`math.nan`:

    - :const:`0` No alleles are associated with this ancestry
    - :const:`1` One allele is associated with this ancestry
    - :const:`2` Both alleles are associated with this ancestry

    Parameters
    ----------
    file_prefix : str
        Path prefix to the set of RFMix files. It will load all of the chromosomes
        at once.
    verbose : bool, optional
        :const:`True` for progress information; :const:`False` otherwise.

    Returns
    -------
    loci : :class:`pandas.DataFrame`
        Loci information for the FB data.
    rf_q : :class:`pandas.DataFrame`
        Global ancestry by chromosome from RFMix.
    admix : :class:`dask.array.Array`
        Local ancestry per population (columns pop1*nsamples ... popX*nsamples).
        This is in order of the populations see `rf_q`.
    """
    from tqdm import tqdm
    from dask.array import concatenate
    # Get file prefixes
    file_prefixes = sorted(glob(file_prefix))
    if len(file_prefixes) == 1:
        file_prefixes = sorted(glob(join(file_prefix, "*")))
    file_prefixes = sorted(_clean_prefixes(file_prefixes))
    fn = [{s: f"{fp}.{s}" for s in ["fb.tsv", "rfmix.Q"]} for fp in file_prefixes]
    # Load loci information
    pbar = tqdm(desc="Mapping loci files", total=len(fn), disable=not verbose)
    loci = _read_file(fn, lambda f: _read_loci(f["fb.tsv"]), pbar)
    pbar.close()
    if len(file_prefixes) > 1 and verbose:
        msg = "Multiple files read in this order:"
        print(f"{msg} {[basename(f) for f in file_prefixes]}")
    # Adjust loci indices and concatenate
    nmarkers = {}
    index_offset = 0
    for i, bi in enumerate(loci):
        nmarkers[fn[i]["fb.tsv"]] = bi.shape[0]
        bi["i"] += index_offset
        index_offset += bi.shape[0]
    loci = concat(loci, axis=0, ignore_index=True)
    # Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping Q files", total=len(fn), disable=not verbose)
    rf_q = _read_file(fn, lambda f: _read_Q(f["rfmix.Q"]), pbar)
    pbar.close()    
    nsamples = rf_q[0].shape[0]
    pops = rf_q[0].drop(["sample_id", "chrom"], axis=1).columns.values
    rf_q = concat(rf_q, axis=0, ignore_index=True)
    # Loading local ancestry by loci
    pbar = tqdm(desc="Mapping fb files", total=len(fn), disable=not verbose)
    admix = _read_file(
        fn,
        lambda f: _read_fb(f["fb.tsv"], nsamples,
                           nmarkers[f["fb.tsv"]], pops, Chunk()),
        pbar,
    )
    pbar.close()
    admix = concatenate(admix, axis=0)    
    return loci, rf_q, admix


def _read_file(fn: List[str], read_func: Callable, pbar=None) -> List:
    """
    Read data from multiple files using a provided read function.

    Parameters:
    ----------
    fn (List[str]): A list of file paths to read.
    read_func (Callable): A function to read data from each file.
    pbar (Optional): A progress bar object to update during reading.

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


def _read_tsv(fn: str) -> DataFrame:
    """
    Read a TSV file into a pandas DataFrame.

    Parameters:
    ----------
    fn (str): File name of the TSV file.

    Returns:
    -------
    DataFrame: DataFrame containing specified columns from the TSV file.
    """
    from numpy import int32
    from pandas import StringDtype
    header = {"chromosome": StringDtype(), "physical_position": int32}
    columns = ["chromosome", "physical_position"]
    try:
        if is_available():
            df = read_csv(
                fn,
                sep="\t",
                header=0,
                usecols=columns,
                dtype=header,
                comment="#"
            )
        else:
            chunks = read_csv(
                fn,
                delim_whitespace=True,
                header=0,
                usecols=columns,
                dtype=header,
                comment="#",
                chunksize=100000, # Low memory chunks
            )        
            # Concatenate chunks into single DataFrame
            df = concat(chunks, ignore_index=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fn} not found.")    
    except Exception as e:
        raise IOError(f"Error reading file {fn}: {e}")    
    # Validate that resulting DataFrame is correct type
    if not isinstance(df, DataFrame):
        raise ValueError(f"Expected a DataFrame but got {type(df)} instead.")    
    # Ensure DataFrame contains correct columns
    if not all(column in df.columns for column in columns):
        raise ValueError(f"DataFrame does not contain expected columns: {columns}")    
    return df


def _read_loci(fn: str) -> DataFrame:
    """
    Read loci information from a TSV file and add a sequential index column.

    Parameters:
    ----------
    fn (str): The file path of the TSV file containing loci information.

    Returns:
    -------
    DataFrame: A pandas DataFrame containing the loci information with an additional 'i' column for indexing.
    """
    df = _read_tsv(fn)
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
        if is_available():
            df = read_csv(
                fn,
                sep="\t",
                header=None,
                names=list(header.keys()),
                dtype=header,
                comment="#"
            )
        else:
            df = read_csv(
                fn,
                delim_whitespace=True,
                header=None,
                names=list(header.keys()),
                dtype=header,
                comment="#",
                compression=None,
                engine="c",
                iterator=False,
            )
    except Exception as e:
        raise IOError(f"Error reading file '{fn}': {e}")
    # Validate that resulting DataFrame is correct type
    if not isinstance(df, DataFrame):
        raise ValueError(f"Expected a DataFrame but got {type(df)} instead.")    
    return df


def _read_Q(fn: str) -> DataFrame:
    """
    Read the Q matrix from a file and add the chromosome information.

    Parameters:
    ----------
    fn (str): The file path of the Q matrix file.

    Returns:
    -------
    DataFrame: The Q matrix with the chromosome information added.
    """
    from re import search
    
    df = _read_Q_noi(fn)
    match = search(r'chr(\d+)', fn)
    if match:
        chrom = match.group(0)
        df["chrom"] = chrom
    else:
        print(f"Warning: Could not extract chromosome information from '{fn}'")        
    return df


def _read_Q_noi(fn: str) -> DataFrame:
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
             chunk: Optional[Chunk] = None) -> Array:
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
    ncols = nsamples * npops * 2
    row_chunk = nrows if chunk.nloci is None else min(nrows, chunk.nloci)
    col_chunk = ncols if chunk.nsamples is None else min(ncols, chunk.nsamples)
    max_npartitions = 16_384
    row_chunk = max(nrows // max_npartitions, row_chunk)
    col_chunk = max(ncols // max_npartitions, col_chunk)
    X = read_fb(fn, nrows, ncols, row_chunk, col_chunk)
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
    from dask.array import concatenate
    
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
    from pandas import StringDtype
    
    try:
        # Read the first two rows of the file, skipping the first row
        if is_available():
            df = read_csv(
                fn,
                sep="\t",
                nrows=2,
                skiprows=1,
            )
        else:
            df = read_csv(
                fn,
                delim_whitespace=True,
                nrows=2,
                skiprows=1,
            )
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
    header = {"sample_id": StringDtype()}
    # Update the header dictionary with the data types of the remaining columns
    header.update(df.dtypes[1:].to_dict())    
    return header


def _clean_prefixes(prefixes):
    """
    Clean and filter a list of file prefixes.

    Parameters:
    ----------
    prefixes (list): A list of file prefixes (paths).

    Returns:
    -------
    list: A list of unique, cleaned file prefixes without the file extensions.

    Notes:
    -----
        - The function removes any prefixes that end with ".logs".
        - It also removes any duplicate prefixes after cleaning.
    """
    cleaned_prefixes = []
    for prefix in prefixes:
        # Split the prefix into directory and base name
        dir_path = dirname(prefix)
        base_name = basename(prefix)
        # Remove the file extensions from the base name
        base = base_name.split(".")[0]        
        # Skip prefixes that end with ".logs"
        if base.startswith("chr"):
            cleaned_prefix = join(dir_path, base)
            cleaned_prefixes.append(cleaned_prefix)
    # Remove duplicate prefixes
    return list(set(cleaned_prefixes))


def set_gpu_environment():
    """
    This function reviews GPU properties.
    """
    from torch.cuda import (
        device_count,
        get_device_properties,
    )
    num_gpus = device_count()
    if num_gpus == 0:
        print("No GPUs available.")
    else:
        for num in range(num_gpus):
            gpu_properties = get_device_properties(num)
            total_memory = gpu_properties.total_memory / (1024 ** 3)
            print(f"GPU {num}: {gpu_properties.name}")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  CUDA capability: {gpu_properties.major}.{gpu_properties.minor}")
