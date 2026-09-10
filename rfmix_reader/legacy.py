"""
Legacy public API, implemented on top of the Dataset core.

Every function here emits a ``DeprecationWarning`` pointing at its
replacement and will be removed in rfmix_reader 1.0.  Results match the
historical ``(loci_df, g_anc, local_array)`` conventions (int8 counts,
ancestries in tool order).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .core.legacy import deprecated, from_legacy

__all__ = [
    "read_rfmix", "read_rfmix_fb", "read_flare", "read_simu",
    "write_data", "admix_to_bed_individual", "generate_tagore_bed",
    "extract_locus_ancestry", "create_binaries",
]


def read_rfmix(file_prefix: str, g_anc: Optional[pd.DataFrame] = None, verbose: bool = True,
               chrom: Optional[str] = None, read_q: bool = True):
    """Deprecated: use :func:`rfmix_reader.open_rfmix` (``ds.la.to_legacy()`` gives this triple)."""
    from .core.api import open_rfmix
    from .readers._common import align_g_anc_columns, maybe_to_backend_frames

    deprecated("read_rfmix", "open_rfmix(...) and ds.la.to_legacy()")
    ds = open_rfmix(file_prefix, source="msp", chrom=chrom, verbose=verbose)
    loci_df, g_out, admix = ds.la.to_legacy()
    if g_anc is not None:
        g_out = align_g_anc_columns(g_anc, ds.la.ancestries)
    elif not read_q:
        g_out = None
    loci_df, g_out = maybe_to_backend_frames(loci_df, g_out)
    return loci_df, g_out, admix


def read_rfmix_fb(file_prefix: str, binary_dir: str = "./binary_files",
                  generate_binary: bool = False, verbose: bool = True,
                  return_original: bool = False, chrom: Optional[str] = None, chunk=None):
    """
    Deprecated: use :func:`rfmix_reader.open_rfmix` with ``source="fb"``.

    ``binary_dir`` is reused as the Zarr ``cache_dir`` when ``generate_binary``
    is true or when it already holds ``<chrom>.zarr`` stores; otherwise the
    Dataset is built in memory.  ``chunk`` is ignored.
    """
    from .core.api import open_rfmix
    from .readers._common import maybe_to_backend_frames

    deprecated("read_rfmix_fb", 'open_rfmix(..., source="fb", cache_dir=...)')
    cache = Path(binary_dir) if binary_dir else None
    use_cache = cache is not None and (generate_binary or any(cache.glob("*.zarr")))
    ds = open_rfmix(file_prefix, source="fb", chrom=chrom, verbose=verbose,
                    keep_posteriors=return_original, cache_dir=cache if use_cache else None)
    loci_df, g_anc, admix = ds.la.to_legacy()
    loci_df, g_anc = maybe_to_backend_frames(loci_df, g_anc)
    if return_original:
        post = ds.la.posterior.data
        X_raw = post.reshape(post.shape[0], -1)
        return loci_df, g_anc, admix, X_raw
    return loci_df, g_anc, admix


def read_flare(file_prefix: str, chunk_size: int = 1_000_000, verbose: bool = True,
               chrom: Optional[str] = None):
    """Deprecated: use :func:`rfmix_reader.open_flare`."""
    from .core.api import open_flare
    from .readers._common import maybe_to_backend_frames

    deprecated("read_flare", "open_flare(...) and ds.la.to_legacy()")
    ds = open_flare(file_prefix, chrom=chrom, verbose=verbose,
                    chunk_rows=max(1, int(chunk_size / 100)))
    loci_df, g_anc, admix = ds.la.to_legacy()
    loci_df, g_anc = maybe_to_backend_frames(loci_df, g_anc)
    return loci_df, g_anc, admix


def read_simu(vcf_path: str, chunk_size: int = 1_000_000, n_threads: int = 16,
              verbose: bool = True, chrom: Optional[str] = None):
    """Deprecated: use :func:`rfmix_reader.open_simu`."""
    from .core.api import open_simu
    from .readers._common import maybe_to_backend_frames

    deprecated("read_simu", "open_simu(...) and ds.la.to_legacy()")
    ds = open_simu(vcf_path, chrom=chrom, verbose=verbose, region_bp=chunk_size,
                   n_threads=n_threads)
    loci_df, g_anc, admix = ds.la.to_legacy()
    loci_df, g_anc = maybe_to_backend_frames(loci_df, g_anc)
    return loci_df, g_anc, admix


def write_data(loci, g_anc, admix, base_rows: int = 100_000, outdir: str = "./output",
               prefix: str = "local-ancestry", verbose: bool = False) -> None:
    """Deprecated: use ``ds.la.to_parquet(outdir, prefix=..., rows_per_file=...)``."""
    deprecated("write_data", "ds.la.to_parquet(...)")
    ds = from_legacy(loci, g_anc, admix)
    ds.la.to_parquet(outdir, prefix=prefix, rows_per_file=base_rows, verbose=verbose)


def admix_to_bed_individual(loci, g_anc, admix, sample_num: int, chunk_size: int = 10_000,
                            min_segment: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Deprecated: use ``ds.la.to_bed(sample, min_segment=...)``."""
    deprecated("admix_to_bed_individual", "ds.la.to_bed(...)")
    ds = from_legacy(loci, g_anc, admix)
    return ds.la.to_bed(int(sample_num), min_segment=min_segment)


def generate_tagore_bed(loci, g_anc, admix, sample_num: int, palette: str = "tab10",
                        chunk_size: int = 10_000, min_segment: int = 3,
                        verbose: bool = True) -> pd.DataFrame:
    """Deprecated: use ``ds.la.to_tagore(sample, palette=..., min_segment=...)``."""
    deprecated("generate_tagore_bed", "ds.la.to_tagore(...)")
    ds = from_legacy(loci, g_anc, admix)
    return ds.la.to_tagore(int(sample_num), palette=palette, min_segment=min_segment)


def extract_locus_ancestry(file_prefix: str, loci: pd.DataFrame, chrom_col: str = "chrom",
                           pos_col: str = "pos", samples: Optional[Sequence[str]] = None,
                           aggregate: bool = True) -> pd.DataFrame:
    """Deprecated: use ``open_rfmix(...).la.at_positions(loci, ...)``."""
    from .core.api import open_rfmix

    deprecated("extract_locus_ancestry", "open_rfmix(...).la.at_positions(...)")
    if loci.empty:
        return loci.copy()
    ds = open_rfmix(file_prefix, source="msp", verbose=False)
    return ds.la.at_positions(loci, chrom_col=chrom_col, pos_col=pos_col,
                              samples=samples, aggregate=aggregate, method="stepwise")


def create_binaries(file_prefix: str, binary_dir: str = "./binary_files",
                    chrom: Optional[str] = None, verbose: bool = True) -> None:
    """Deprecated: use :func:`rfmix_reader.convert` (``rfmix-reader convert fb ...``)."""
    from .utils import create_binaries as _impl

    deprecated("create_binaries", 'convert(path, "fb", cache_dir)',
               "The .bin files are only needed by the deprecated readers.read_rfmix module.")
    _impl(file_prefix, binary_dir, chrom=chrom, verbose=verbose)
