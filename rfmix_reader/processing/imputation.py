"""
Functions to impute loci to genotype.

This is a time consuming process, but should only need to be done once.
Loading the data becomes very fast because data is saved to a Zarr.
"""
from __future__ import annotations

import logging
import zarr
import warnings
import numpy as np
from tqdm import tqdm
from pandas import DataFrame
from typing import Literal, Optional, TYPE_CHECKING

from ..backends import _select_array_backend

if TYPE_CHECKING:
    from dask.array import Array

InterpMethod = Literal["linear", "nearest", "stepwise"]

GPU_ENABLED: bool = _select_array_backend().__name__ == "cupy"


def _sentinel_to_nan(batch) -> np.ndarray:
    """Readers mark missing calls with ``-1`` (int8); the Zarr uses NaN."""
    out = np.asarray(batch, dtype=np.float32)
    if out.size and (out < 0).any():
        out = out.copy()
        out[out < 0] = np.nan
    return out


def _to_host(x):
    """Convert an array-module array back to a NumPy array on host."""
    if hasattr(x, "__cuda_array_interface__") and hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


logger = logging.getLogger(__name__)


def _print_logger(message: str) -> None:
    """Log a progress message."""
    logger.info(message)


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
    col_imputed : same type as the selected array backend
        Interpolated values, either float (posteriors) or int (hard states).
    """
    mod = _select_array_backend()
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
        # Round to nearest integer to produce hard ancestry calls (0/1/2).
        # This preserves RFMix semantics for downstream GWAS but discards
        # posterior uncertainty. Use method='nearest' or 'stepwise' to
        # assign observed values directly without rounding.
        out[mask] = mod.round(interp_vals).astype(mod.float32)
        return out

    idx = mod.arange(n, dtype=mod.int64)

    if method == "stepwise":
        left_idx = mod.where(valid, idx, -1)
        # maximum.accumulate propagates the last-seen valid index forward.
        # Positions before the first valid point still hold -1.
        left_nearest = mod.maximum.accumulate(left_idx)
        idx_valid_or_n = mod.where(valid, idx, n)
        first_valid = mod.min(idx_valid_or_n)
        # Replace pre-first-valid -1 entries with first_valid (forward fill from
        # left boundary). Trailing NaN positions already have a valid left_nearest
        # (the last valid index), so they are correctly forward-filled too.
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

    Notes
    -----
    Columns with no missing values are skipped entirely before interpolation
    begins, which avoids redundant computation for fully-observed loci.
    """
    mod = _select_array_backend()
    block = mod.asarray(block, dtype=mod.float32)
    loci_dim, sample_dim, ancestry_dim = block.shape

    flat = block.reshape(loci_dim, -1)  # (loci, samples*ancestries)
    x = mod.asarray(pos, dtype=mod.float32) if pos is not None else None
    # Pre-compute NaN mask per column to skip columns that need no interpolation
    has_nan = mod.isnan(flat).any(axis=0)
    for j in range(flat.shape[1]):
        if not has_nan[j]:
            continue
        flat[:, j] = _interpolate_1d(flat[:, j], x=x, method=method)

    return flat.reshape(loci_dim, sample_dim, ancestry_dim)


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
    admix : dask.array.Array (loci, samples, ancestries)
        Dask array containing the local ancestry data to be stored in the Zarr
        array.
    zarr_outdir : str
        Directory path where the Zarr array will be saved.
    batch_size : int
        Batch size for processing local ancestry data. Default is 10,000.

    Returns
    -------
    zarr.Array (variants, samples, ancestries)
        The populated Zarr array containing the expanded local ancestry data
        with NaNs.

    Notes
    -----
    - The resulting Zarr array is saved to disk at the specified path.
    - If the admix array fits within 50% of available system memory it is
      pre-materialized in one ``compute()`` call; otherwise data is read in
      contiguous slices per batch to reduce redundant Dask I/O. The 50%
      threshold can be a concern for full 22-chromosome datasets concatenated
      in memory (~2–3 GB); lower it or call per-chromosome to reduce peak RSS.
    - The non-NaN ``"i"`` values in ``variant_loci_df`` must be monotonically
      non-decreasing (i.e., the DataFrame must be sorted by genomic position)
      for the contiguous-slice optimization to produce correct results.
    """
    _print_logger("Generate empty Zarr.")
    n_samples = min(500, admix.shape[1])
    z = zarr.open(f"{zarr_outdir}/local-ancestry.zarr", mode="w",
                  shape=(variant_loci_df.shape[0],
                         admix.shape[1], admix.shape[2]),
                  chunks=(8000, n_samples, admix.shape[2]),
                  dtype='float32', fill_value=np.nan)

    if "i" not in variant_loci_df.columns:
        raise ValueError(
            "variant_loci_df must contain column 'i' mapping "
            "variant rows to admix rows"
        )

    i = variant_loci_df["i"].to_numpy()
    dest_idx = np.flatnonzero(~np.isnan(i))
    src_idx  = i[dest_idx].astype(np.int64)

    if src_idx.size > 1 and not (np.diff(src_idx) >= 0).all():
        raise ValueError(
            "The 'i' column of variant_loci_df is not monotonically non-decreasing "
            "at non-NaN positions. Ensure variant_loci_df is sorted by genomic "
            "position before calling _expand_array."
        )

    # Pre-materialize admix if it fits in available memory; otherwise use
    # contiguous slice access (src_idx is sorted) to avoid expensive Dask
    # fancy indexing per batch.
    from psutil import virtual_memory
    admix_bytes = admix.nbytes
    avail_mem = virtual_memory().available
    admix_np = None
    if admix_bytes < avail_mem * 0.5:
        _print_logger("Pre-materializing admix array (fits in memory).")
        admix_np = admix.compute()

    # Add admix into zarr
    _print_logger("Filling Zarr with local ancestry data in batches.")
    for start in range(0, dest_idx.size, batch_size):
        end = min(start + batch_size, dest_idx.size)

        # Write only valid rows to avoid overwriting NaNs
        d = dest_idx[start:end]
        batch_src = src_idx[start:end]
        if admix_np is not None:
            z[d, :, :] = _sentinel_to_nan(admix_np[batch_src])
        else:
            # Use contiguous slice + local reindex to avoid Dask fancy indexing.
            # Use min/max so this is correct even if src_idx has repeated values.
            lo, hi = int(batch_src.min()), int(batch_src.max()) + 1
            slab = admix[lo:hi].compute()
            z[d, :, :] = _sentinel_to_nan(slab[batch_src - lo])

    _print_logger("Zarr array successfully populated!")
    return z


def interpolate_array(
    variant_loci_df: DataFrame, admix: Array, zarr_outdir: str,
    chunk_size: int = 50_000, batch_size: int = 10_000,
    interpolation: InterpMethod | str = "linear", use_bp_positions: bool = False,
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
        If True, use ``variant_loci_df['pos']`` as the x-axis for interpolation,
        weighting gaps by physical distance. If False, loci are treated as
        equally spaced (index-based), which is inaccurate across regions of
        variable window density such as centromeres and telomeres. Prefer
        ``True`` whenever variant positions span centromeric gaps.

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

    Examples
    --------
    >>> import pandas as pd
    >>> import dask.array as da
    >>> variant_loci_df = pd.DataFrame({'chrom': ['1', '1'], 'pos': [100, 200]})
    >>> admix = da.random.random((2, 2, 3))
    >>> z = interpolate_array(variant_loci_df, admix, '/path/to/output',
                              chunk_size=1, interpolation='linear')
    >>> print(z.shape)
    (2, 2, 3)
    """
    method = _normalize_method(interpolation)

    pos = None
    if use_bp_positions:
        if "pos" not in variant_loci_df.columns:
            raise ValueError(
                "use_bp_positions=True but 'pos' column not found in variant_loci_df."
            )
        pos = variant_loci_df["pos"].to_numpy(dtype=np.float32)
        if len(pos) > 1 and not (np.diff(pos) >= 0).all():
            raise ValueError(
                "variant_loci_df must be sorted by 'pos' in ascending order. "
                "Call .sort_values('pos').reset_index(drop=True) before passing."
            )

    _print_logger("Starting expansion!")
    z = _expand_array(variant_loci_df, admix, zarr_outdir,
                      batch_size=batch_size)

    total_rows, _, _ = z.shape
    _print_logger(f"Interpolating data using method='{method}'!")

    mod = _select_array_backend()
    remaining = np.zeros((z.shape[1], z.shape[2]), dtype=bool)
    # Rows that carry observed data (non-NaN 'i'); used to give every chunk the
    # nearest observed row before and after it as context, so the result does
    # not depend on chunk_size and no chunk starts or ends inside a gap.
    observed = np.flatnonzero(~np.isnan(variant_loci_df["i"].to_numpy(dtype=float)))
    for start in tqdm(range(0, total_rows, chunk_size),
                      desc="Interpolating chunks", unit="chunk"):
        end = min(start + chunk_size, total_rows)
        k = np.searchsorted(observed, start)
        prev_row = int(observed[k - 1]) if k > 0 else None
        k2 = np.searchsorted(observed, end)
        next_row = int(observed[k2]) if k2 < observed.size else None
        lead = 1 if prev_row is not None else 0
        parts = ([z[prev_row:prev_row + 1]] if lead else []) + [z[start:end]] + \
                ([z[next_row:next_row + 1]] if next_row is not None else [])
        chunk = mod.array(np.concatenate(parts, axis=0), dtype=mod.float32)
        pos_chunk = None
        if pos is not None:
            pos_parts = ([pos[prev_row:prev_row + 1]] if lead else []) + [pos[start:end]] + \
                        ([pos[next_row:next_row + 1]] if next_row is not None else [])
            pos_chunk = np.concatenate(pos_parts)
        if start == 0 and not mod.isnan(chunk).any():
            warnings.warn(
                "No NaNs detected in first chunk; interpolation may be unnecessary."
            )
        interp_chunk = interpolate_block(chunk, method=method, pos=pos_chunk)[lead:lead + (end - start)]
        remaining |= _to_host(mod.isnan(interp_chunk).any(axis=0))
        z[start:end, :, :] = _to_host(interp_chunk)

    # Gaps longer than a chunk cannot be filled chunk-locally: interpolate the
    # affected (sample, ancestry) columns once more over the full locus axis.
    if remaining.any():
        _fill_remaining_gaps(z, remaining, method=method, pos=pos)

    _print_logger("Interpolation complete!")
    return z


def _fill_remaining_gaps(
    z, remaining: np.ndarray, *, method: InterpMethod, pos: Optional[np.ndarray],
    sample_batch: int = 64,
) -> None:
    """
    Second interpolation pass for columns that still contain NaN.

    ``remaining`` is a boolean ``(samples, ancestries)`` mask.  Affected samples
    are read in batches across *all* loci, interpolated, and written back.
    Columns with no valid value at all are left as NaN with a warning.
    """
    mod = _select_array_backend()
    samples = np.flatnonzero(remaining.any(axis=1))
    _print_logger(
        f"Filling gaps longer than one chunk for {samples.size} sample(s)."
    )
    unfilled = 0
    for start in range(0, samples.size, sample_batch):
        idx = samples[start:start + sample_batch]
        block = mod.asarray(z.oindex[:, idx, :], dtype=mod.float32)
        block = interpolate_block(block, method=method, pos=pos)
        unfilled += int(_to_host(mod.isnan(block).any(axis=0)).sum())
        z.oindex[:, idx, :] = _to_host(block)
    if unfilled:
        warnings.warn(
            f"{unfilled} (sample, ancestry) column(s) have no observed value and "
            "remain NaN after interpolation."
        )
