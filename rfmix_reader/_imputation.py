"""
Functions to imputate loci to genotype.

This is a time consuming process, but should only need to be done once.
Loading the data becomes very fast because data is saved to a Zarr.
"""
import zarr
from tqdm import tqdm
from time import strftime
from pandas import DataFrame
from dask.array import Array

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    import cupy as arr_mod
else:
    import numpy as arr_mod

__all__ = [
    "interpolate_array",
    "_interpolate_col",
    "_expand_array",
    "_print_logger"
]

def _print_logger(message: str) -> None:
    """
    Print a timestamped log message to the console.

    This function prepends the current date and time to the provided message
    and prints it to the console. It's designed for simple logging purposes
    within a program.

    Parameters
    ----------
    message : str
        The message to be logged. This should be a string containing the
        information you want to log.

    Returns
    -------
    None
        This function doesn't return any value; it prints the log message
        directly to the console.
    """
    current_time = strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}")


def _expand_array(
        variant_loci_df: DataFrame, admix: Array, zarr_outdir: str
) -> zarr.Array:
    """
    Expand and fill a Zarr array with local ancestry data, handling missing
    values.

    This function creates a Zarr array based on the shape of input DataFrames,
    fills it with NaN values where data is missing, and then populates it with
    local ancestry data where available.

    Parameters
    ----------
    variant_loci_df : pandas.DataFrame
        DataFrame containing the data to be expanded. Used to determine the
        shape of the output array and identify missing data.
    admix : dask.array.Array
        Dask array containing the local ancestry data to be stored in the Zarr
        array.
    zarr_outdir : str
        Directory path where the Zarr array will be saved.

    Returns
    -------
    zarr.Array
        The populated Zarr array containing the expanded local ancestry data
        with NANs.

    Notes
    -----
    - The resulting Zarr array is saved to disk at the specified path.
    - Memory usage may be high when dealing with large datasets.
    """
    from numpy import array, nan, int32
    _print_logger("Generate empty Zarr!")
    z = zarr.open(f"{zarr_outdir}/local-ancestry.zarr", mode="w",
                  shape=(variant_loci_df.shape[0], admix.shape[1]),
                  chunks=(8000, 2000), dtype='float32')

    # Fill with NaNs
    arr_nans = array(variant_loci_df.loc[variant_loci_df.isnull()\
                                         .any(axis=1)].index, dtype=int32)
    _print_logger("Fill Zarr with NANs!")
    z[arr_nans, :] = nan
    _print_logger("Remove NaN array!")
    del arr_nans

    # Fill with local ancestry
    arr = array(variant_loci_df.dropna().index)
    _print_logger("Fill Zarr with data!")
    z[arr, :] = admix.compute()
    return z


def _interpolate_col(col: arr_mod.ndarray) -> arr_mod.ndarray:
    """
    Interpolate missing values in a column of data.

    This function performs linear interpolation on missing values (NaNs) in the
    input column. If CUDA is available, it uses cupy for GPU-accelerated
    computation; otherwise, it falls back to numpy.

    Parameters
    ----------
    col : ndarray
        A 1D numpy or cupy array representing a column of data, potentially
        containing NaN values.

    Returns
    -------
    ndarray
        A copy of the input array with NaN values interpolated and rounded to
        the nearest integer.
    """
    mask = arr_mod.isnan(col)
    idx = arr_mod.arange(len(col))
    valid = ~mask

    if arr_mod.any(valid):
        interpolated = arr_mod.round(arr_mod.interp(idx[mask], idx[valid],
                                                    col[valid]))
        col = col.copy() # Avoid modifying the original array
        col[mask] = interpolated.astype(int)
    return col


def interpolate_array(
        variant_loci_df: DataFrame, admix: Array, zarr_outdir: str,
        chunk_size: int = 50000) -> zarr.Array:
    """
    Interpolate missing values in a large array of genetic data.

    This function expands the input data into a Zarr array and then performs
    column-wise interpolation on chunks of the data to fill in missing values.

    Parameters
    ----------
    variant_loci_df : pandas.DataFrame
        DataFrame containing variant and loci information.
    admix : dask.array.Array
        Dask array containing the admixture data to be interpolated.
    zarr_outdir : str
        Directory path where the Zarr array will be saved.
    chunk_size : int, optional
        Number of rows to process in each chunk. Default is 50000.

    Returns
    -------
    zarr.Array
        The Zarr array containing the interpolated data.

    Notes
    -----
    - This function uses CUDA acceleration if available, otherwise falls back to
      NumPy.
    - The function processes the data in chunks to manage memory usage for large
      datasets.
    - Progress is displayed using a tqdm progress bar.
    - The interpolation is performed column-wise using the `_interpolate_col`
      function.

    Examples
    --------
    >>> import pandas as pd
    >>> import dask.array as da
    >>> variant_loci_df = pd.DataFrame({'chrom': ['1', '1'], 'pos': [100, 200]})
    >>> admix = da.random.random((2, 3))
    >>> z = interpolate_array(variant_loci_df, admix, '/path/to/output',
                              chunk_size=1)
    >>> print(z.shape)
    (2, 3)
    """
    _print_logger("Starting expansion!")
    z = _expand_array(variant_loci_df, admix, zarr_outdir)
    total_rows, _ = z.shape

    # Process the data in chunks
    _print_logger("Interpolating data!")
    for i in tqdm(range(0, total_rows, chunk_size),
                  desc="Processing chunks", unit="chunk"):
        end = min(i + chunk_size, total_rows)
        chunk = arr_mod.array(z[i:end, :])
        interp_chunk = arr_mod.apply_along_axis(_interpolate_col, axis=0,
                                                arr=chunk)
        z[i:end, :] = interp_chunk.get() if is_available() else interp_chunk

    return z


def _load_genotypes(plink_prefix_path):
    from tensorqtl import pgen
    pgr = pgen.PgenReader(plink_prefix_path)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df


def _load_admix(prefix_path, binary_dir):
    from rfmix_reader import read_rfmix
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def __testing__():
    basename = "/projects/b1213/large_projects/brain_coloc_app/input"
    # Local ancestry
    prefix_path = f"{basename}/local_ancestry_rfmix/_m/"
    binary_dir = f"{basename}/local_ancestry_rfmix/_m/binary_files/"
    loci, rf_q, admix = _load_admix(prefix_path, binary_dir)
    loci.rename(columns={"chromosome": "chrom",
                         "physical_position": "pos"},
                inplace=True)
    sample_ids = list(rf_q.sample_id.unique().to_pandas())
    # Variant data
    plink_prefix = f"{basename}/genotypes/TOPMed_LIBD"
    _, variant_df = _load_genotypes(plink_prefix)
    variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
                                            keep='first')
    variant_loci_df = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
                          how="outer", indicator=True)\
                   .loc[:, ["chrom", "pos", "i"]]
    data_path = f"{basename}/local_ancestry_rfmix/_m"
    z = interpolate_array(variant_loci_df, admix, data_path)
