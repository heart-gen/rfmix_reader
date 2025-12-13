"""
Functions to imputate loci to genotype.

This is a time consuming process, but should only need to be done once.
Loading the data becomes very fast because data is saved to a Zarr.
"""
from __future__ import annotations

import zarr
import numpy as np
import warnings
from tqdm import tqdm
from time import strftime
from pandas import DataFrame
from dask.array import Array
from typing import Literal, Optional

try:
    from torch.cuda import is_available as cuda_is_available
except ModuleNotFoundError:
    def cuda_is_available() -> bool:
        return False

try:
    import cupy as cp
except ImportError:
    cp = None

GPU_ENABLED = bool(cuda_is_available() and (cp is not None))

if GPU_ENABLED:
    arr_mod = cp
else:
    arr_mod = np

InterpMethod = Literal["linear", "nearest", "stepwise"]

def _to_host(x):
    """Convert an array-module array back to a NumPy array on host."""
    if GPU_ENABLED and hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


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


def _normalize_method(method: str) -> str:
    m = method.lower()
    if m in ("linear", "lin"):
        return "linear"
    if m in ("nearest", "midpoint", "nearest_neighbor", "nearest-neighbor"):
        return "nearest"
    if m in ("step", "stepwise", "nearest_segment", "nearest-segment"):
        return "stepwise"
    raise ValueError(f"Unknown interpolation method: {method}")


def _interpolate_1d(
    col, x: Optional[np.ndarray] = None, method: InterpMethod = "linear"
):
    """
    Interpolate a 1D ancestry trajectory.

    Parameters
    ----------
    col : array-like
        Shape (n_loci,), dtype float, with NaNs for missing loci.
    x : array-like, optional
        Monotonic coordinate along loci (e.g., bp position). If None,
        the index (0..n_loci-1) is used.
    method : {"linear","nearest","stepwise"}

    Returns
    -------
    col_imputed : same type as arr_mod
        Interpolated values, either float (posteriors) or int (hard states).
    """
    mod = arr_mod
    col = mod.asarray(col, dtype=mod.float32)
    mask = mod.isnan(col)
    if not bool(mask.any()):
        return col

    n = int(col.shape[0])
    xp = mod.arange(n, dtype=mod.float32) if x is None else mod.asarray(x, dtype=mod.float32)
    valid = ~mask
    if not bool(valid.any()):
        return col # All NaNs, nothing to impute

    method = _normalize_method(method)

    if method == "linear":
        xp_valid = xp[valid]
        y_valid = col[valid]
        xp_nan = xp[mask]
        interp_vals = mod.interp(xp_nan, xp_valid, y_valid)
        out = col.copy()
        # Final ancestry is hard (0/1). Rounding to preserve RFMix semantics
        out[mask] = mod.round(interp_vals).astype(mod.float32)
        return out

    idx = mod.arange(n, dtype=mod.int64)

    if method == "stepwise":
        left_idx = mod.where(valid, idx, -1)
        left_nearest = mod.maximum.accumulate(left_idx)
        idx_valid_or_n = mod.where(valid, idx, n)
        first_valid = mod.min(idx_valid_or_n)
        left_nearest = mod.where(left_nearest < 0, first_valid, left_nearest)
        out = col.copy()
        out[mask] = col[left_nearest[mask]]
        return out

    if method == "nearest":
        left_idx = mod.where(valid, idx, -1)
        left_nearest = mod.maximum.accumulate(left_idx)
        right_idx_pre = mod.where(valid, idx, n)
        right_nearest = mod.minimum.accumulate(right_idx_pre[::-1])[::-1]
        xp_all = xp

        safe_left = mod.where(left_nearest < 0, 0, left_nearest)
        safe_right = mod.where(right_nearest >= n, n - 1, right_nearest)
        left_pos = xp_all[safe_left]
        right_pos = xp_all[safe_right]

        dist_left = mod.where(left_nearest >= 0, xp_all - left_pos, float("inf"))
        dist_right = mod.where(right_nearest < n, right_pos - xp_all, float("inf"))
        nearest_idx = mod.where(dist_left <= dist_right, left_nearest, right_nearest)
        out = col.copy()
        out[mask] = col[nearest_idx[mask]]
        return out

    raise ValueError(f"Unexpected interpolation method after normalization: {method}")


def interpolate_block(
    block, *, method: InterpMethod = "linear", pos: Optional[np.ndarray] = None,
):
    """
    Block-wise interpolation for a haplotype / ancestry block.

    `method` can be "linear", "nearest", or "stepwise". If `pos` is given,
    interpolation is performed in bp space; otherwise it is done in index
    space (0..n_loci-1).

    Returns a float32 array in the same array module (NumPy or CuPy).
    """
    mod = arr_mod
    block = mod.asarray(block, dtype=mod.float32)
    loci_dim, sample_dim, ancestry_dim = block.shape

    flat = block.reshape(loci_dim, -1)  # (loci, samples*ancestries)
    x = mod.asarray(pos, dtype=mod.float32) if pos is not None else None
    for j in range(flat.shape[1]):
        flat[:, j] = _interpolate_1d(flat[:, j], x=x, method=method)

    return flat.reshape(loci_dim, sample_dim, ancestry_dim)


def _interpolate_col(col):
    """
    Backwards-compatible shim: original code interpolated a single
    column linearly.

    TODO: deprecate
    """
    return _interpolate_1d(col, method="linear")


def _expand_array(
    variant_loci_df: DataFrame, admix: Array, zarr_outdir: str,
    batch_size: int = 10_000
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
    batch_size : int
        Batch size for processing local ancestry data. Default is 10,000.

    Returns
    -------
    zarr.Array
        The populated Zarr array containing the expanded local ancestry data
        with NaNs.

    Notes
    -----
    - The resulting Zarr array is saved to disk at the specified path.
    - Memory usage may be high when dealing with large datasets.
    """
    _print_logger("Generate empty Zarr.")
    z = zarr.open(f"{zarr_outdir}/local-ancestry.zarr", mode="w",
                  shape=(variant_loci_df.shape[0],
                         admix.shape[1], admix.shape[2]),
                  chunks=(8000, 200, admix.shape[2]),
                  dtype='float32', fill_value=np.nan)

    nan_rows_mask = variant_loci_df.isnull().any(axis=1).values
    nan_rows_mask = arr_mod.asarray(nan_rows_mask)
    nan_indices = arr_mod.where(nan_rows_mask)[0]
    nan_indices = _to_host(nan_indices)

    # Materialize admix blocks in manageable batches
    _print_logger("Filling Zarr with local ancestry data in batches.")
    for start in range(0, admix.shape[0], batch_size):
        end = min(start + batch_size, admix.shape[0])
        valid = ~nan_rows_mask[start:end]
        if not valid.any():
            continue

        # Write only valid rows to avoid overwriting NaNs
        z_slice = z[start:end]
        z_slice[valid] = admix[start:end][valid].compute()

    _print_logger("Zarr array successfully populated!")
    return z


def interpolate_array(
    variant_loci_df: DataFrame, admix: Array, zarr_outdir: str,
    chunk_size: int = 50_000, batch_size: int = 10_000,
    interpolation: str = "linear", use_bp_positions: bool = False,
) -> zarr.Array:
    """
    Interpolate missing local ancestry entries on the variant grid.

    This function expands the input data into a Zarr array and then performs
    column-wise interpolation on chunks of the data to fill in missing values.

    Parameters
    ----------
    variant_loci_df : pandas.DataFrame
        DataFrame defining the variant grid and missing loci.
        Must be sorted by genomic coordinate; should contain at least:
           - 'chrom' (optional but nice for debugging)
           - 'pos'   (used when `use_bp_positions=True`)
    admix : dask.array.Array
        Local ancestry array with shape (loci, samples, ancestries).
    zarr_outdir : str
        Directory path where the Zarr array will be saved.
    chunk_size : int, optional
        Number of variant rows to interpolate per chunk. Default is 50,000.
    batch_size : int
        Batch size for processing local ancestry data. Default is 10,000.
    interpolation : {"linear","nearest","stepwise"}, default "linear"
        Interpolation scheme.
    use_bp_positions : bool, default False
        If True, use `variant_loci_df['pos']` as the x-axis for interpolation.
        If False, loci are treated as equally spaced (index-based).

    Returns
    -------
    zarr.Array
        Zarr-backed (variants, samples, ancestries) array with missing rows imputed.

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
                              chunk_size=1, interpolation='linear')
    >>> print(z.shape)
    (2, 3)
    """
    method = _normalize_method(interpolation)

    _print_logger("Starting expansion!")
    z = _expand_array(variant_loci_df, admix, zarr_outdir,
                      batch_size=batch_size)

    total_rows, _, _ = z.shape
    _print_logger(f"Interpolating data using method='{method}'!")

    pos = None
    if use_bp_positions:
        if "pos" not in variant_loci_df.columns:
            raise ValueError("use_bp_positions=True but 'pos' column not found in variant_loci_df.")
        pos = variant_loci_df["pos"].to_numpy(dtype=np.float32)

    for start in tqdm(range(0, total_rows, chunk_size),
                      desc="Interpolating chunks", unit="chunk"):
        end = min(start + chunk_size, total_rows)
        chunk = arr_mod.array(z[start:end, :, :], dtype=arr_mod.float32)
        pos_chunk = None if pos is None else pos[start:end]
        if not arr_mod.isnan(chunk).any():
            warnings.warn(
                f"No NaNs detected in chunk {start}:{end}; interpolation skipped."
            )
        interp_chunk = interpolate_block(chunk, method=method, pos=pos_chunk)
        z[start:end, :, :] = _to_host(interp_chunk)

    _print_logger("Interpolation complete!")
    return z
