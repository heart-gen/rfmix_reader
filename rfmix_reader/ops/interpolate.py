"""Interpolate local ancestry onto a denser variant grid (per chromosome, Zarr-backed)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from ..core import schema as S
from ..formats.common import normalize_chrom_label

__all__ = ["interpolate", "build_variant_grid"]


def build_variant_grid(ds: xr.Dataset, variants: pd.DataFrame, *, chrom_col: str = "chrom",
                       pos_col: str = "pos", include_source: bool = True) -> pd.DataFrame:
    """
    Merge target ``variants`` with the Dataset's own variants into the
    ``variant_loci_df`` grid used by :func:`interpolate_array`.

    Returns a frame with ``chrom``, ``pos`` and ``i`` (index into the Dataset,
    NaN where the position is not a source variant), sorted by chromosome
    (Dataset order) then position.  Only chromosomes present in the Dataset
    are kept.
    """
    src_chrom = np.array([str(c) for c in ds[S.CHROMOSOME].values])
    src_norm = np.array([normalize_chrom_label(c) for c in src_chrom])
    src_pos = np.asarray(ds[S.VARIANT_POSITION].values, dtype=np.int64)
    source = pd.DataFrame({"_norm": src_norm, "chrom": src_chrom, "pos": src_pos,
                           "i": np.arange(len(src_pos), dtype=float)})

    tgt = pd.DataFrame({
        "_norm": [normalize_chrom_label(str(c)) for c in variants[chrom_col]],
        "pos": pd.to_numeric(variants[pos_col], errors="raise").astype(np.int64),
    }).drop_duplicates()
    tgt = tgt[tgt["_norm"].isin(set(src_norm))]

    frames = []
    for norm in dict.fromkeys(src_norm.tolist()):
        s = source[source["_norm"] == norm]
        t = tgt[tgt["_norm"] == norm]
        label = s["chrom"].iloc[0]
        merged = t.merge(s[["pos", "i"]], on="pos", how="outer" if include_source else "left")
        merged["chrom"] = label
        merged = merged.sort_values("pos").drop_duplicates("pos")
        frames.append(merged[["chrom", "pos", "i"]])
    if not frames:
        raise ValueError("None of the requested chromosomes are present in the Dataset.")
    return pd.concat(frames, ignore_index=True)


def interpolate(
    ds: xr.Dataset, variants: pd.DataFrame, zarr_outdir, *, chrom_col: str = "chrom",
    pos_col: str = "pos", method: str = "linear", use_bp_positions: bool = True,
    chunk_size: int = 50_000, batch_size: int = 10_000, include_source: bool = True,
    max_chunk_bytes: int | None = None,
) -> xr.DataArray:
    """
    Interpolate diploid counts onto the positions in ``variants``.

    One Zarr array is written per chromosome under
    ``<zarr_outdir>/<chrom>/local-ancestry.zarr`` (see
    :func:`rfmix_reader.processing.imputation.interpolate_array` for the
    methods); the result is returned as a lazy ``(variant, sample, ancestry)``
    float32 DataArray with ``chromosome`` / ``variant_position`` coordinates.
    Missing calls (``-1``) are treated as gaps and filled.

    ``max_chunk_bytes`` caps the bytes one interpolation chunk may hold; ``None``
    (the default here) uses the budget of
    :func:`rfmix_reader.processing.imputation.interpolate_array`.

    The Dataset's own variants are the interpolation anchors, so they are
    always part of the grid handed to the imputer; ``include_source`` selects
    only what is *returned* (``False``: one row per requested position;
    ``True``: the union of requested and source positions).  The Zarr arrays
    on disk therefore always hold the full interpolation grid, which is a
    superset of the returned rows when ``include_source=False``; address the
    result by its ``variant_position`` coordinate rather than by row offset.
    """
    import dask.array as da
    from ..processing.imputation import interpolate_array

    # Interpolation is only defined relative to the observed calls, so the grid
    # handed to the imputer always carries every source variant as an anchor.
    # Requesting positions that fall between source variants with
    # ``include_source=False`` used to drop those anchors, leaving the imputer
    # to interpolate between whichever requested positions happened to
    # coincide with a variant.
    grid = build_variant_grid(ds, variants, chrom_col=chrom_col, pos_col=pos_col,
                              include_source=True)
    requested: dict[str, np.ndarray] | None = None
    if not include_source:
        rows = build_variant_grid(ds, variants, chrom_col=chrom_col, pos_col=pos_col,
                                  include_source=False)
        requested = {str(label): sub["pos"].to_numpy(dtype=np.int64)
                     for label, sub in rows.groupby("chrom", sort=False)}
    zarr_outdir = Path(zarr_outdir)
    counts = ds.la.counts.data

    arrays: List = []
    chroms: List[np.ndarray] = []
    positions: List[np.ndarray] = []
    for label in dict.fromkeys(grid["chrom"].tolist()):
        want = None if requested is None else requested.get(str(label), np.empty(0, dtype=np.int64))
        if want is not None and want.size == 0:
            continue                        # nothing requested on this chromosome
        g = grid[grid["chrom"] == label].reset_index(drop=True)
        outdir = zarr_outdir / str(label)
        outdir.mkdir(parents=True, exist_ok=True)
        # the imputer indexes `i` into the array it is given: pass the whole
        # counts array (lazy) so global indices stay valid
        kw = {} if max_chunk_bytes is None else {"max_chunk_bytes": max_chunk_bytes}
        z = interpolate_array(
            g, counts, str(outdir), chunk_size=chunk_size, batch_size=batch_size,
            interpolation=method, use_bp_positions=use_bp_positions, **kw,
        )
        arr = da.from_array(z, chunks=z.chunks)
        pos_out = g["pos"].to_numpy(dtype=np.int64)
        if want is not None:                # return the requested rows only
            keep = np.flatnonzero(np.isin(pos_out, want))
            arr, pos_out = arr[keep], pos_out[keep]
        arrays.append(arr)
        chroms.append(np.array([label] * len(pos_out), dtype=object))
        positions.append(pos_out.astype(np.int32))

    data = da.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    return xr.DataArray(
        data, dims=(S.VARIANT, S.SAMPLE, S.ANCESTRY),
        coords={
            S.CHROMOSOME: (S.VARIANT, np.concatenate(chroms)),
            S.VARIANT_POSITION: (S.VARIANT, np.concatenate(positions)),
            S.SAMPLE_ID: ds[S.SAMPLE_ID],
            S.ANCESTRY: ds[S.ANCESTRY],
        },
        name="local_ancestry",
    )
