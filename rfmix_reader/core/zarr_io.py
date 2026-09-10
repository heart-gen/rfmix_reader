"""
Streaming writer / lazy opener for the per-chromosome Zarr cache.

A store is written once by :func:`write_store` from a parser's chunk stream
(constant memory: one chunk in flight) and reopened lazily with
:func:`open_store` as an :class:`xarray.Dataset` following :mod:`core.schema`.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import xarray as xr
import zarr

from ..formats.base import Chunk, Header
from ..formats.global_ancestry import fractions_from_counts
from . import schema as S

__all__ = ["write_store", "open_store", "store_path"]


def store_path(cache_dir, chrom: str) -> Path:
    """``<cache_dir>/<chrom>.zarr``."""
    return Path(cache_dir) / f"{chrom}.zarr"


def _string_array(group: zarr.Group, name: str, values, dims) -> zarr.Array:
    arr = group.create_array(name, shape=(len(values),), dtype=str, dimension_names=dims)
    arr[:] = np.asarray([str(v) for v in values], dtype=object)
    return arr


def write_store(
    header: Header, chunks: Iterable[Chunk], path, *,
    chunk_rows: int = 10_000, overwrite: bool = True,
    source_format: str = "unknown", verbose: bool = False,
) -> Path:
    """
    Stream ``chunks`` into a new Zarr store at ``path``.

    Parameters
    ----------
    header : Header
        Parser header (samples, ancestries, optional global ancestry).
    chunks : iterable of Chunk
        Variant blocks in genomic order.
    path : str or Path
        Destination store (a directory).  Removed first when ``overwrite``.
    chunk_rows : int
        Zarr chunk length along ``variant``.
    source_format : str
        Recorded in the store attributes.
    """
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists (pass overwrite=True).")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    S_, A = header.n_samples, header.n_ancestries
    root = zarr.open_group(str(path), mode="w")

    hap = root.create_array(
        S.HAPLOTYPE_ANCESTRY, shape=(0, S_, 2), chunks=(chunk_rows, S_, 2),
        dtype="int8", dimension_names=(S.VARIANT, S.SAMPLE, S.PLOIDY),
    )
    pos = root.create_array(S.VARIANT_POSITION, shape=(0,), chunks=(chunk_rows,),
                            dtype="int32", dimension_names=(S.VARIANT,))
    end = root.create_array(S.SEGMENT_END, shape=(0,), chunks=(chunk_rows,),
                            dtype="int32", dimension_names=(S.VARIANT,))
    chrom = root.create_array(S.CHROMOSOME, shape=(0,), chunks=(chunk_rows,),
                              dtype=str, dimension_names=(S.VARIANT,))
    post = None
    if header.has_posterior:
        post = root.create_array(
            S.POSTERIOR, shape=(0, S_, 2, A), chunks=(chunk_rows, S_, 2, A),
            dtype="float32", dimension_names=(S.VARIANT, S.SAMPLE, S.PLOIDY, S.ANCESTRY),
        )

    _string_array(root, S.SAMPLE_ID, header.samples, (S.SAMPLE,))
    _string_array(root, S.ANCESTRY, header.ancestries, (S.ANCESTRY,))
    ploidy = root.create_array(S.PLOIDY, shape=(2,), dtype="int8", dimension_names=(S.PLOIDY,))
    ploidy[:] = np.array([0, 1], dtype=np.int8)

    hap_counts = np.zeros((S_, A), dtype=np.int64)
    n_rows = 0
    contig_label: Optional[str] = header.chrom
    iterator = chunks
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(chunks, desc=f"Writing {path.name}", unit="chunk")
    for chunk in iterator:
        n = len(chunk)
        if n == 0:
            continue
        codes = np.asarray(chunk.codes, dtype=np.int8)
        if codes.shape != (n, S_, 2):
            raise ValueError(f"chunk codes have shape {codes.shape}; expected {(n, S_, 2)}.")
        hap.append(codes, axis=0)
        pos.append(np.asarray(chunk.pos, dtype=np.int32), axis=0)
        end.append(np.asarray(chunk.end, dtype=np.int32), axis=0)
        chrom.append(np.asarray(chunk.chrom, dtype=object), axis=0)
        if post is not None:
            if chunk.posterior is None:
                raise ValueError("Parser promised posteriors but a chunk has none.")
            post.append(np.asarray(chunk.posterior, dtype=np.float32), axis=0)
        if header.global_ancestry is None:
            valid = codes >= 0
            for a in range(A):
                hap_counts[:, a] += ((codes == a) & valid).sum(axis=(0, 2))
        if contig_label is None:
            contig_label = str(chunk.chrom[0])
        n_rows += n

    if contig_label is None:
        contig_label = "unknown"
    if header.global_ancestry is not None:
        ga = np.asarray(header.global_ancestry, dtype=np.float32)
    else:
        ga = fractions_from_counts(hap_counts)
    ga_arr = root.create_array(
        S.GLOBAL_ANCESTRY, shape=(1, S_, A), dtype="float32",
        dimension_names=(S.CONTIG, S.SAMPLE, S.ANCESTRY),
    )
    ga_arr[:] = ga[None, :, :]
    _string_array(root, S.CONTIG, [contig_label], (S.CONTIG,))

    hap.attrs["coordinates"] = " ".join(
        [S.CHROMOSOME, S.VARIANT_POSITION, S.SEGMENT_END, S.SAMPLE_ID, S.ANCESTRY]
    )
    root.attrs.update({
        "source_format": source_format,
        "source_files": [str(f) for f in header.source_files],
        "rfmix_reader_version": S._version(),
        "n_variants": int(n_rows),
    })
    return path


def open_store(path) -> xr.Dataset:
    """Open one cache store lazily as a schema-conforming Dataset."""
    from . import accessor  # noqa: F401  (registers ``ds.la``)

    ds = xr.open_zarr(str(path), consolidated=False)
    coords = [c for c in (S.CHROMOSOME, S.VARIANT_POSITION, S.SEGMENT_END, S.SAMPLE_ID,
                          S.ANCESTRY, S.PLOIDY, S.CONTIG) if c in ds.variables]
    ds = ds.set_coords(coords)
    return S.validate(ds)
