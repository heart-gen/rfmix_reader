"""
Dataset-returning readers.

``open_rfmix`` / ``open_flare`` / ``open_simu`` parse a source once — either
into an in-memory Dataset or, when ``cache_dir`` is given, into a
per-chromosome Zarr store that is reopened lazily on every later call.
``open_local_ancestry`` opens an existing cache; ``convert`` only writes it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from ..formats.discover import discover, get_parser, primary_file
from ..formats.base import Header, chrom_label_from_path
from . import schema as S
from .zarr_io import open_store, store_path, write_store

__all__ = ["open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _chrom_label(fmt: str, filemap: Dict[str, str], header: Header) -> str:
    label = chrom_label_from_path(primary_file(fmt, filemap)) or header.chrom
    if label is None:
        raise ValueError(
            f"Cannot infer a chromosome label for {primary_file(fmt, filemap)}; "
            "name files with the chromosome (e.g. run_chr21.msp.tsv)."
        )
    return label


def _build_in_memory(fmt: str, filemap: Dict[str, str], header: Header,
                     chunk_rows: int, opts: Dict[str, object]) -> xr.Dataset:
    parser = get_parser(fmt)
    chroms, positions, ends, codes, posts = [], [], [], [], []
    for chunk in parser.iter_chunks(filemap, header, chunk_rows, **opts):
        if len(chunk) == 0:
            continue
        chroms.append(np.asarray(chunk.chrom, dtype=str))
        positions.append(chunk.pos)
        ends.append(chunk.end)
        codes.append(chunk.codes)
        if header.has_posterior:
            posts.append(chunk.posterior)
    if not codes:
        raise ValueError(f"No variants found in {primary_file(fmt, filemap)}.")

    hap = np.concatenate(codes, axis=0)
    global_anc = header.global_ancestry
    if global_anc is None:
        from ..formats.global_ancestry import fractions_from_counts

        valid = hap >= 0
        counts = np.stack(
            [((hap == a) & valid).sum(axis=(0, 2)) for a in range(header.n_ancestries)],
            axis=1,
        )
        global_anc = fractions_from_counts(counts)

    return S.build_dataset(
        np.concatenate(chroms), np.concatenate(positions), np.concatenate(ends), hap,
        header.samples, header.ancestries,
        posterior=np.concatenate(posts, axis=0) if posts else None,
        global_ancestry=global_anc, contig=[header.chrom or str(chroms[0][0])],
        source_format=fmt, source_files=header.source_files, chunk_rows=chunk_rows,
    )


def _open_one(fmt: str, filemap: Dict[str, str], *, cache_dir, overwrite: bool,
              chunk_rows: int, verbose: bool, opts: Dict[str, object]) -> xr.Dataset:
    parser = get_parser(fmt)
    if cache_dir is None:
        header = parser.scan(filemap, **opts)
        return _build_in_memory(fmt, filemap, header, chunk_rows, opts)

    label = chrom_label_from_path(primary_file(fmt, filemap))
    header = None
    if label is None:
        header = parser.scan(filemap, **opts)
        label = _chrom_label(fmt, filemap, header)
    path = store_path(cache_dir, label)
    if path.exists() and not overwrite:
        if verbose:
            logger.info("Opening cached %s", path)
        return open_store(path)

    header = header or parser.scan(filemap, **opts)
    if verbose:
        logger.info("Converting %s -> %s", primary_file(fmt, filemap), path)
    write_store(
        header, parser.iter_chunks(filemap, header, chunk_rows, **opts), path,
        chunk_rows=chunk_rows, overwrite=True, source_format=fmt, verbose=verbose,
    )
    return open_store(path)


def _open_format(fmt: str, path: str, *, chrom, cache_dir, overwrite, chunk_rows,
                 verbose, opts) -> xr.Dataset:
    from . import accessor  # noqa: F401  (registers ``ds.la``)

    filemaps = discover(path, fmt, chrom)
    datasets = [
        _open_one(fmt, fm, cache_dir=cache_dir, overwrite=overwrite,
                  chunk_rows=chunk_rows, verbose=verbose, opts=opts)
        for fm in filemaps
    ]
    return S.concat_datasets(datasets)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def open_rfmix(
    path: str, *, source: str = "msp", chrom: Optional[str] = None,
    cache_dir=None, keep_posteriors: bool = False, overwrite: bool = False,
    chunk_rows: int = 10_000, verbose: bool = True,
) -> xr.Dataset:
    """
    Open RFMix output as a local-ancestry Dataset.

    Parameters
    ----------
    path : str
        Directory, file, or path prefix of the RFMix outputs.
    source : {"msp", "fb"}
        ``"msp"`` reads ``.msp.tsv`` segments (default, fast); ``"fb"`` reads
        the ``.fb.tsv`` posteriors (one streaming pass, no ``.bin`` step).
    chrom : str, optional
        Restrict to one chromosome.
    cache_dir : str or Path, optional
        Write/reuse ``<cache_dir>/<chrom>.zarr``.  Without it the Dataset is
        built in memory (fine for ``msp``; for ``fb`` consider a cache).
    keep_posteriors : bool
        ``fb`` only: also store the float32 posteriors (``ds.la.posterior``).
    overwrite : bool
        Rebuild existing stores.
    chunk_rows : int
        Variants per chunk (parsing and Zarr chunking).
    verbose : bool
        Progress logging.

    Returns
    -------
    xarray.Dataset
        See :mod:`rfmix_reader.core.schema`; use ``ds.la`` for views.
    """
    if source not in ("msp", "fb"):
        raise ValueError("source must be 'msp' or 'fb'.")
    if source == "fb" and keep_posteriors and cache_dir is None:
        import warnings

        warnings.warn(
            "keep_posteriors=True without cache_dir holds the full float32 posterior "
            "matrix in memory (e.g. ~11 GB for a chr1-sized cohort of 500 samples); "
            "pass cache_dir to stream it to Zarr instead.", stacklevel=2,
        )
    opts = {"keep_posteriors": keep_posteriors} if source == "fb" else {}
    return _open_format(source, path, chrom=chrom, cache_dir=cache_dir,
                        overwrite=overwrite, chunk_rows=chunk_rows, verbose=verbose,
                        opts=opts)


def open_flare(
    path: str, *, chrom: Optional[str] = None, cache_dir=None,
    overwrite: bool = False, chunk_rows: int = 10_000, verbose: bool = True,
) -> xr.Dataset:
    """Open FLARE ``.anc.vcf.gz`` (+ ``.global.anc.gz``) output as a Dataset."""
    return _open_format("flare", path, chrom=chrom, cache_dir=cache_dir,
                        overwrite=overwrite, chunk_rows=chunk_rows, verbose=verbose,
                        opts={})


def open_simu(
    path: str, *, chrom: Optional[str] = None, cache_dir=None,
    overwrite: bool = False, chunk_rows: int = 10_000, region_bp: int = 1_000_000,
    n_threads: int = 4, verbose: bool = True,
) -> xr.Dataset:
    """Open haptools ``simgenotype`` VCFs (``POP`` field) as a Dataset."""
    return _open_format("haptools", path, chrom=chrom, cache_dir=cache_dir,
                        overwrite=overwrite, chunk_rows=chunk_rows, verbose=verbose,
                        opts={"region_bp": region_bp, "n_threads": n_threads})


def open_local_ancestry(cache_dir, chrom: Optional[str] = None) -> xr.Dataset:
    """
    Open an existing cache directory (``<chrom>.zarr`` stores) lazily.

    Stores are concatenated along ``variant`` in chromosome order.
    """
    from . import accessor  # noqa: F401
    from ..formats.common import chrom_sort_key, normalize_chrom_label

    cache_dir = Path(cache_dir)
    stores = sorted((p for p in cache_dir.glob("*.zarr") if p.is_dir()),
                    key=lambda p: chrom_sort_key(p.stem))
    if chrom is not None:
        target = normalize_chrom_label(str(chrom))
        stores = [p for p in stores if normalize_chrom_label(p.stem) == target]
    if not stores:
        raise FileNotFoundError(f"No Zarr stores found in {cache_dir}"
                                + (f" for chromosome '{chrom}'" if chrom else "") + ".")
    return S.concat_datasets([open_store(p) for p in stores])


def convert(
    path: str, fmt: str, cache_dir, *, chrom: Optional[str] = None,
    keep_posteriors: bool = False, overwrite: bool = False,
    chunk_rows: int = 10_000, n_threads: int = 4, verbose: bool = True,
) -> List[Path]:
    """
    Convert a source into ``<cache_dir>/<chrom>.zarr`` stores (one per
    chromosome) and return their paths.  Existing stores are skipped unless
    ``overwrite``.
    """
    parser = get_parser(fmt)
    opts: Dict[str, object] = {}
    if fmt == "fb":
        opts["keep_posteriors"] = keep_posteriors
    if fmt == "haptools":
        opts["n_threads"] = n_threads

    written = []
    for filemap in discover(path, fmt, chrom):
        header = None
        label = chrom_label_from_path(primary_file(fmt, filemap))
        if label is None:
            header = parser.scan(filemap, **opts)
            label = _chrom_label(fmt, filemap, header)
        target = store_path(cache_dir, label)
        if target.exists() and not overwrite:
            if verbose:
                logger.info("Skipping existing %s", target)
            written.append(target)
            continue
        header = header or parser.scan(filemap, **opts)
        write_store(
            header, parser.iter_chunks(filemap, header, chunk_rows, **opts), target,
            chunk_rows=chunk_rows, overwrite=True, source_format=fmt, verbose=verbose,
        )
        written.append(target)
    return written
