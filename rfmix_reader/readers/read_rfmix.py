"""
Adapted from `_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py
"""
from __future__ import annotations
import warnings
import logging
from re import search
from tqdm import tqdm
from glob import glob
from numpy import int32, array
from collections import OrderedDict as odict
from typing import Optional, List, Tuple, Dict
from os.path import basename, dirname, join, exists

from .fb_read import read_fb
from ..io import BinaryFileNotFoundError, Chunk
from ..utils import (
    _read_file,
    get_prefixes,
    create_binaries,
    set_gpu_environment,
    filter_file_maps_by_chrom,
)
from ..backends import _select_dataframe_backend, _select_array_backend

logger = logging.getLogger(__name__)


def _get_dataframe_backend():
    return _select_dataframe_backend()


def _get_array_backend():
    return _select_array_backend()

def read_rfmix(
        file_prefix: str, binary_dir: str = "./binary_files",
        generate_binary: bool = False, verbose: bool = True,
        return_hap_matrix: bool = False,
        return_original: bool = False,
        chrom: Optional[str] = None,
) -> (
    Tuple[DataFrame, DataFrame, Array]
    | Tuple[DataFrame, DataFrame, Array, Array]
):
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
    return_hap_index : bool, optional
        Return the haplotypes index for reconstruction of hap0 / hap1.
    return_original : bool, optional
        Return the original RFMix matrix ``X_raw`` along with the summed
        ancestry counts.
    chrom : str, optional
        If provided, restrict reading to a single chromosome whose label
        matches ``chrom`` (with or without the ``chr`` prefix).

    Returns
    -------
    loci_df : :class:`pandas.DataFrame`
        Loci information for the FB data.
    g_anc : :class:`pandas.DataFrame`
        Global ancestry by chromosome from RFMix.
    local_array : :class:`dask.array.Array`
        Local ancestry per population stacked (variants, samples, ancestries).
        This is in order of the populations see `g_anc`.
    X_raw : :class:`dask.array.Array`, optional
        Returned only when ``return_original`` is :const:`True`. The unphased
        RFMix matrix prior to haplotype summarization.
    return_hap_matrix : bool
        Whether to return local ancestry with haplotypes (hap0 / hap1) level
        information.

    Notes
    -----
    Local ancestry output will be either :const:`0`, :const:`1`, :const:`2`, or
    :data:`math.nan`:

    - :const:`0` No alleles are associated with this ancestry
    - :const:`1` One allele is associated with this ancestry
    - :const:`2` Both alleles are associated with this ancestry
    """
    from dask.array import Array, concatenate, stack

    df_mod = _get_dataframe_backend()
    concat = df_mod.concat
    array_mod = _get_array_backend()
    use_gpu = array_mod.__name__ == "cupy" or df_mod.__name__ == "cudf"

    # Device information
    if verbose and use_gpu:
        try:
            set_gpu_environment()
        except ModuleNotFoundError:
            logger.warning("PyTorch not available; skipping GPU environment info.")

    # Get file prefixes
    fn = filter_file_maps_by_chrom(
        get_prefixes(file_prefix, "rfmix", verbose), chrom, kind="RFMix"
    )

    # Load loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn), disable=not verbose)
    loci_dfs = _read_file(fn, lambda f: _read_loci(f["fb.tsv"]), pbar)
    pbar.close()

    # Adjust loci indices and concatenate
    nmarkers = {}; index_offset = 0; loci_by_fn = {}
    for i, bi in enumerate(loci_dfs):
        nmarkers[fn[i]["fb.tsv"]] = bi.shape[0]
        bi["i"] += index_offset
        index_offset += bi.shape[0]
        loci_by_fn[fn[i]["fb.tsv"]] = bi

    loci_df = concat(loci_dfs, axis=0, ignore_index=True)

    # Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn),
                disable=not verbose)
    g_anc = _read_file(fn, lambda f: _read_Q(f["rfmix.Q"]), pbar)
    pbar.close()

    nsamples = g_anc[0].shape[0]
    pops = g_anc[0].drop(["sample_id", "chrom"], axis=1).columns.values
    g_anc = concat(g_anc, axis=0, ignore_index=True)

    # Loading local ancestry by loci
    if generate_binary:
        create_binaries(file_prefix, binary_dir)

    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn),
                disable=not verbose)
    local_data = _read_file(
        fn,
        lambda f: _read_fb(
            f["fb.tsv"], nsamples, nmarkers[f["fb.tsv"]], pops,
            binary_dir, Chunk(),
        ),
        pbar,
    )
    pbar.close()

    # Unpack data
    admix_list = [admix for admix, X_raw in local_data]
    X_raw_list = [X_raw for admix, X_raw in local_data]

    # Stack across chromosomes
    local_array = concatenate(admix_list, axis=0)
    if return_original:
        X_raw = concatenate(X_raw_list, axis=0)
        return loci_df, g_anc, local_array, X_raw
    return loci_df, g_anc, local_array


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
    df_mod = _get_dataframe_backend()
    DataFrame = df_mod.DataFrame
    read_csv = df_mod.read_csv
    concat = df_mod.concat
    CategoricalDtype = df_mod.CategoricalDtype
    header = {"chromosome": CategoricalDtype(), "physical_position": int32}
    try:
        if df_mod.__name__ == "cudf":
            df = read_csv(fn, sep="\t", header=0, usecols=list(header.keys()),
                          dtype=header, comment="#", compression="infer")
        else:
            chunks = read_csv(
                fn, sep=r"\s+", header=0, usecols=list(header.keys()),
                dtype=header, comment="#", compression="infer",
                chunksize=100_000, # Low memory chunks
            )
            # Concatenate chunks into single DataFrame
            df = concat(chunks, ignore_index=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fn} not found.")
    except Exception as e:
        raise OSError(f"Error reading file {fn}: {e}") from e

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
    df_mod = _get_dataframe_backend()
    DataFrame = df_mod.DataFrame
    read_csv = df_mod.read_csv
    try:
        if df_mod.__name__ == "cudf":
            df = read_csv(fn, sep="\t", header=None, names=list(header.keys()),
                          dtype=header, comment="#")
        else:
            df = read_csv(fn, sep=r"\s+", header=None,
                          names=list(header.keys()), dtype=header, comment="#",
                          compression=None, engine="c", iterator=False)
    except Exception as e:
        raise OSError(f"Error reading file {fn}: {e}") from e

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
    df = _read_Q_noi(fn)

    m = search(r'chr[\d]+', fn)
    if m:
        chrom = m.group(0)
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
        raise OSError(f"Error reading file {fn}: {e}") from e


def _read_fb(
    fn: str, nsamples: int, nloci: int, pops: list, temp_dir: str,
    chunk: Optional[Chunk] = None,
) -> Tuple[Array, Array]:
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
    tuple
        The summed forward-backward matrix and the original raw matrix.
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
    admix = _subset_populations(X, npops)

    return admix, X


def _subset_populations(X: Array, npops: int) -> Array:
    """
    Subset and process the input array X based on populations.

    Parameters:
    X (dask.array): Input array where columns represent data for different populations.
    npops (int): Number of populations for column processing.

    Returns:
    admix_summed : dask.array.Array
        Processed array with adjacent columns summed for each population subset.
    """
    import numpy as np
    ncols = X.shape[1]
    if ncols % npops != 0:
        raise ValueError("The number of columns in X must be divisible by npops.")

    if ncols % (2 * npops) != 0:
        raise ValueError(
            "The number of columns in X must be divisible by (2 * npops). "
            "Expected layout: 2 haplotypes per sample per ancestry."
        )

    nsamples = ncols // (2 * npops)
    pop_subset = []

    for pop_start in range(npops):
        X0 = X[:, pop_start::npops] # Subset based on populations
        if int(X0.shape[1]) % 2 != 0:
            raise ValueError("Number of columns must be even.")

        X0_summed = X0[:, ::2] + X0[:, 1::2] # Sum adjacent columns
        pop_subset.append(X0_summed)

    return stack(pop_subset, axis=2)


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
    df_mod = _get_dataframe_backend()
    DataFrame = df_mod.DataFrame
    read_csv = df_mod.read_csv
    CategoricalDtype = df_mod.CategoricalDtype
    try:
        # Read the first two rows of the file, skipping the first row
        if df_mod.__name__ == "cudf":
            df = read_csv(fn, sep="\t", nrows=2, skiprows=1)
        else:
            df = read_csv(fn, sep=r"\s+", nrows=2, skiprows=1)

    except FileNotFoundError:
        raise FileNotFoundError(f"File '{fn}' not found.")
    except Exception as e:
        raise OSError(f"Error reading file {fn}: {e}") from e

    # Validate that the resulting DataFrame is of the correct type
    if not isinstance(df, DataFrame):
        raise ValueError(f"Expected a DataFrame but got {type(df)} instead.")
    # Ensure the DataFrame contains at least one column
    if df.shape[1] < 1:
        raise ValueError("The DataFrame does not contain any columns.")

    # Initialize the header dictionary with the sample_id column
    header = {"sample_id": CategoricalDtype()}
    # Update the header dictionary with the data types of the remaining columns
    header.update(df.dtypes[1:].to_dict())
    return header


# Convenience: expose helper utilities on the main reader function to make
# them easy to reach for tests and advanced users who rely on the original
# script-style API.
read_rfmix._read_tsv = _read_tsv
read_rfmix._read_csv = _read_csv
read_rfmix._read_Q = _read_Q
read_rfmix._read_Q_noi = _read_Q_noi
read_rfmix._subset_populations = _subset_populations
read_rfmix._read_fb = _read_fb
read_rfmix.BinaryFileNotFoundError = BinaryFileNotFoundError
