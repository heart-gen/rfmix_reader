"""
The local-ancestry Dataset schema shared by every reader, the Zarr cache and
``processing.phase``.

.. code-block:: text

    dims:   variant, sample, ploidy (=2), ancestry, contig
    vars:   haplotype_ancestry (variant, sample, ploidy)           int8   code into `ancestry`; -1 missing
            posterior          (variant, sample, ploidy, ancestry) float32  optional
            global_ancestry    (contig, sample, ancestry)          float32
    coords: chromosome (variant) str, variant_position (variant) int32,
            segment_end (variant) int32, sample_id (sample) str,
            ancestry (ancestry) str [tool order], ploidy [0, 1], contig (contig) str
    attrs:  source_format, source_files, rfmix_reader_version
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import xarray as xr

from ..readers._common import MISSING

__all__ = [
    "MISSING",
    "VARIANT", "SAMPLE", "PLOIDY", "ANCESTRY", "CONTIG",
    "HAPLOTYPE_ANCESTRY", "POSTERIOR", "GLOBAL_ANCESTRY",
    "CHROMOSOME", "VARIANT_POSITION", "SEGMENT_END", "SAMPLE_ID",
    "build_dataset", "validate", "concat_datasets",
]

VARIANT = "variant"
SAMPLE = "sample"
PLOIDY = "ploidy"
ANCESTRY = "ancestry"
CONTIG = "contig"

HAPLOTYPE_ANCESTRY = "haplotype_ancestry"
POSTERIOR = "posterior"
GLOBAL_ANCESTRY = "global_ancestry"

CHROMOSOME = "chromosome"
VARIANT_POSITION = "variant_position"
SEGMENT_END = "segment_end"
SAMPLE_ID = "sample_id"

VARIANT_COORDS = (CHROMOSOME, VARIANT_POSITION, SEGMENT_END)


def _version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def build_dataset(
    chromosome, variant_position, segment_end, haplotype_ancestry,
    sample_id: Sequence[str], ancestry: Sequence[str],
    *, posterior=None, global_ancestry=None, contig: Optional[Sequence[str]] = None,
    source_format: str = "unknown", source_files: Optional[Iterable[str]] = None,
    chunk_rows: int = 10_000,
) -> xr.Dataset:
    """
    Assemble an in-memory (dask-backed) Dataset following the schema.

    ``haplotype_ancestry`` is ``(variant, sample, 2)`` int8; ``posterior`` is
    ``(variant, sample, 2, ancestry)`` float32 or None; ``global_ancestry`` is
    ``(contig, sample, ancestry)`` float32 (a 2-D ``(sample, ancestry)`` array
    is accepted for a single contig).
    """
    import dask.array as da

    sample_id = [str(s) for s in sample_id]
    ancestry = [str(a) for a in ancestry]
    S, A = len(sample_id), len(ancestry)

    hap = da.asarray(haplotype_ancestry)
    if hap.ndim != 3 or hap.shape[1] != S or hap.shape[2] != 2:
        raise ValueError(
            f"haplotype_ancestry must be (variant, {S}, 2); got {hap.shape}."
        )
    hap = hap.astype(np.int8).rechunk((chunk_rows, S, 2))
    L = int(hap.shape[0])

    chromosome = np.asarray(chromosome).astype(str)
    variant_position = np.asarray(variant_position, dtype=np.int32)
    segment_end = (variant_position if segment_end is None
                   else np.asarray(segment_end, dtype=np.int32))
    for name, arr in ((CHROMOSOME, chromosome), (VARIANT_POSITION, variant_position),
                      (SEGMENT_END, segment_end)):
        if arr.shape != (L,):
            raise ValueError(f"{name} must have shape ({L},); got {arr.shape}.")

    if contig is None:
        contig = list(dict.fromkeys(chromosome.tolist())) or ["unknown"]
    contig = [str(c) for c in contig]

    data_vars = {
        HAPLOTYPE_ANCESTRY: ((VARIANT, SAMPLE, PLOIDY), hap),
    }
    if posterior is not None:
        post = da.asarray(posterior)
        if post.shape != (L, S, 2, A):
            raise ValueError(
                f"posterior must be ({L}, {S}, 2, {A}); got {post.shape}."
            )
        data_vars[POSTERIOR] = (
            (VARIANT, SAMPLE, PLOIDY, ANCESTRY),
            post.astype(np.float32).rechunk((chunk_rows, S, 2, A)),
        )
    if global_ancestry is not None:
        ga = np.asarray(global_ancestry, dtype=np.float32)
        if ga.ndim == 2:
            ga = ga[None, :, :]
        if ga.shape != (len(contig), S, A):
            raise ValueError(
                f"global_ancestry must be ({len(contig)}, {S}, {A}); got {ga.shape}."
            )
        data_vars[GLOBAL_ANCESTRY] = ((CONTIG, SAMPLE, ANCESTRY), ga)

    coords = {
        CHROMOSOME: (VARIANT, chromosome),
        VARIANT_POSITION: (VARIANT, variant_position),
        SEGMENT_END: (VARIANT, segment_end),
        SAMPLE_ID: (SAMPLE, np.asarray(sample_id, dtype=object)),
        ANCESTRY: (ANCESTRY, np.asarray(ancestry, dtype=object)),
        PLOIDY: (PLOIDY, np.array([0, 1], dtype=np.int8)),
        CONTIG: (CONTIG, np.asarray(contig, dtype=object)),
    }
    attrs = {
        "source_format": source_format,
        "source_files": [str(f) for f in (source_files or [])],
        "rfmix_reader_version": _version(),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def validate(ds: xr.Dataset) -> xr.Dataset:
    """Raise ``ValueError`` if ``ds`` does not follow the schema; return it otherwise."""
    if HAPLOTYPE_ANCESTRY not in ds:
        raise ValueError(f"Dataset has no '{HAPLOTYPE_ANCESTRY}' variable.")
    hap = ds[HAPLOTYPE_ANCESTRY]
    if tuple(hap.dims) != (VARIANT, SAMPLE, PLOIDY):
        raise ValueError(
            f"'{HAPLOTYPE_ANCESTRY}' must have dims {(VARIANT, SAMPLE, PLOIDY)}, "
            f"got {tuple(hap.dims)}."
        )
    if hap.dtype != np.int8:
        raise ValueError(f"'{HAPLOTYPE_ANCESTRY}' must be int8, got {hap.dtype}.")
    if ds.sizes[PLOIDY] != 2:
        raise ValueError(f"ploidy must be 2, got {ds.sizes[PLOIDY]}.")
    for coord, dim in ((CHROMOSOME, VARIANT), (VARIANT_POSITION, VARIANT),
                       (SAMPLE_ID, SAMPLE), (ANCESTRY, ANCESTRY)):
        if coord not in ds.coords:
            raise ValueError(f"Dataset is missing coordinate '{coord}'.")
        if tuple(ds[coord].dims) != (dim,):
            raise ValueError(f"Coordinate '{coord}' must have dims ({dim},).")
    if POSTERIOR in ds:
        post = ds[POSTERIOR]
        if tuple(post.dims) != (VARIANT, SAMPLE, PLOIDY, ANCESTRY):
            raise ValueError(
                f"'{POSTERIOR}' must have dims {(VARIANT, SAMPLE, PLOIDY, ANCESTRY)}."
            )
    if GLOBAL_ANCESTRY in ds:
        ga = ds[GLOBAL_ANCESTRY]
        if tuple(ga.dims) != (CONTIG, SAMPLE, ANCESTRY):
            raise ValueError(
                f"'{GLOBAL_ANCESTRY}' must have dims {(CONTIG, SAMPLE, ANCESTRY)}."
            )
    return ds


def concat_datasets(datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Concatenate per-chromosome Datasets along ``variant`` (and ``contig``).

    Samples and ancestries must be identical, in the same order.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets to concatenate.")
    if len(datasets) == 1:
        return datasets[0]

    ref_samples = datasets[0][SAMPLE_ID].values.tolist()
    ref_anc = datasets[0][ANCESTRY].values.tolist()
    for ds in datasets[1:]:
        if ds[SAMPLE_ID].values.tolist() != ref_samples:
            raise ValueError("Sample sets differ between per-chromosome datasets.")
        if ds[ANCESTRY].values.tolist() != ref_anc:
            raise ValueError("Ancestry labels differ between per-chromosome datasets.")

    variant_vars = [HAPLOTYPE_ANCESTRY] + ([POSTERIOR] if all(POSTERIOR in d for d in datasets) else [])
    variant_part = xr.concat(
        [d[variant_vars] for d in datasets], dim=VARIANT,
        data_vars="minimal", coords="minimal", compat="override",
    )
    out = variant_part
    if all(GLOBAL_ANCESTRY in d for d in datasets):
        ga = xr.concat([d[GLOBAL_ANCESTRY] for d in datasets], dim=CONTIG)
        out = out.assign({GLOBAL_ANCESTRY: ga})
    out.attrs = dict(datasets[0].attrs)
    out.attrs["source_files"] = [f for d in datasets for f in d.attrs.get("source_files", [])]
    return out
