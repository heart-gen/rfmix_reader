"""
Adapted from `_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py
"""
import warnings
from glob import glob
from dask.array import Array
from collections import OrderedDict as odict
from os.path import basename, dirname, join, exists
from typing import Optional, Callable, List, Tuple, Dict

from ._chunk import Chunk
from ._fb_read import read_fb
from ._utils import set_gpu_environment
from ._utils import get_prefixes, create_binaries
from ._errorhandling import BinaryFileNotFoundError

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, read_csv, concat
else:
    from pandas import DataFrame, read_csv, concat

__all__ = ["read_rfmix"]

def read_rfmix(
        file_prefix: str, binary_dir: str = "./binary_files",
        generate_binary: bool = False, verbose: bool = True,
) -> Tuple[DataFrame, DataFrame, Array]:
    """
    Read RFMix files into data frames and a Dask array.

    Parameters
    ----------
    file_prefix : str
        Path prefix to the set of RFMix files. It will load all of the chromosomes
        at once.
    binary_dir : str, optional
        Path prefix to the binary version of RFMix (*fb.tsv) files. Default is
        "./binary_files".
    generate_binary: bool, optional
       :const:`True` generate the binary file. Default: `False`.
    verbose : bool, optional
        :const:`True` for progress information; :const:`False` otherwise.
        Default:`True`.

    Returns
    -------
    loci : :class:`pandas.DataFrame`
        Loci information for the FB data.
    rf_q : :class:`pandas.DataFrame`
        Global ancestry by chromosome from RFMix.
    admix : :class:`dask.array.Array`
        Local ancestry per population (columns pop1*nsamples ... popX*nsamples).
        This is in order of the populations see `rf_q`.

    Notes
    -----
    Local ancestry output will be either :const:`0`, :const:`1`, :const:`2`, or
    :data:`math.nan`:

    - :const:`0` No alleles are associated with this ancestry
    - :const:`1` One allele is associated with this ancestry
    - :const:`2` Both alleles are associated with this ancestry
    """
    from tqdm import tqdm
    from dask.array import concatenate
    # Device information
    if verbose and is_available():
        set_gpu_environment()
    # Get file prefixes
    fn = get_prefixes(file_prefix, verbose)
    # Load loci information
    pbar = tqdm(desc="Mapping loci files", total=len(fn), disable=not verbose)
    loci = _read_file(fn, lambda f: _read_loci(f["fb.tsv"]), pbar)
    pbar.close()
    # Adjust loci indices and concatenate
    nmarkers = {}; index_offset = 0
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
    if generate_binary:
        create_binaries(file_prefix, binary_dir)
    pbar = tqdm(desc="Mapping fb files", total=len(fn), disable=not verbose)
    admix = _read_file(
        fn,
        lambda f: _read_fb(f["fb.tsv"], nsamples,
                           nmarkers[f["fb.tsv"]], pops,
                           binary_dir, Chunk()),
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
    try:
        if is_available():
            df = read_csv(
                fn,
                sep="\t",
                header=0,
                usecols=list(header.keys()),
                dtype=header,
                comment="#"
            )
        else:
            ## TODO: FutureWarning: The 'delim_whitespace' keyword in
            ## pd.read_csv is deprecated and will be removed in a future
            ## version. Use ``sep='\s+'`` instead.

            chunks = read_csv(
                fn,
                delim_whitespace=True,
                header=0,
                usecols=list(header.keys()),
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
    if not all(column in df.columns for column in list(header.keys())):
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
    DataFrame: A DataFrame containing the loci information with an
               additional 'i' column for indexing.
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
