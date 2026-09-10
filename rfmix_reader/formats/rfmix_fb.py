"""
RFMix ``.fb.tsv`` parser (forward-backward posteriors), single streaming pass.

Each chunk of rows is parsed with the pandas C engine, reshaped to
``(rows, samples, 2, ancestries)``, and reduced to per-haplotype argmax codes.
Posteriors are kept only when ``keep_posteriors=True``.
"""
from __future__ import annotations

import gzip
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from ..readers._common import MISSING, align_g_anc_columns
from ..readers.read_rfmix import _read_fb_pops
from .base import Chunk, Header, chrom_label_from_path
from .global_ancestry import frame_to_array, read_rfmix_q

FORMAT = "fb"
_META_COLS = 4


def discover(path: str, chrom: Optional[str] = None):
    from .discover import discover as _discover

    return _discover(path, FORMAT, chrom)


def _column_names(fn: str) -> List[str]:
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt") as fh:
        fh.readline()
        return fh.readline().rstrip("\n").split("\t")


def _samples_from_columns(cols: List[str], pops: List[str]) -> List[str]:
    """Sample names in file order; validates the sample-major/hap/pop layout."""
    data_cols = cols[_META_COLS:]
    stride = 2 * len(pops)
    if len(data_cols) % stride:
        raise ValueError(
            f"{len(data_cols)} data columns is not a multiple of 2 x {len(pops)} populations."
        )
    samples = []
    for i in range(0, len(data_cols), stride):
        block = data_cols[i:i + stride]
        parts = [c.split(":::") for c in block]
        names = {p[0] for p in parts}
        if len(names) != 1:
            raise ValueError(f"Columns {block[0]}..{block[-1]} mix samples.")
        expected = [f"{parts[0][0]}:::hap{h}:::{p}" for h in (1, 2) for p in pops]
        if block != expected:
            raise ValueError(
                f"Unexpected column layout at {block[0]}: expected {expected}."
            )
        samples.append(parts[0][0])
    return samples


def scan(filemap: Dict[str, str], keep_posteriors: bool = False, **_) -> Header:
    fn = filemap["fb.tsv"]
    pops = _read_fb_pops(fn)
    samples = _samples_from_columns(_column_names(fn), pops)

    global_anc = None
    if "rfmix.Q" in filemap:
        q = align_g_anc_columns(read_rfmix_q(filemap["rfmix.Q"], add_chrom=False), pops)
        global_anc = frame_to_array(q, samples, pops)

    return Header(
        samples=samples, ancestries=pops, source_files=sorted(filemap.values()),
        chrom=chrom_label_from_path(fn), n_variants=None, global_ancestry=global_anc,
        has_posterior=keep_posteriors,
    )


def _codes_from_posteriors(post: np.ndarray) -> np.ndarray:
    """``(n, S, 2, A)`` posteriors -> ``(n, S, 2)`` int8 argmax codes, -1 if no mass."""
    codes = post.argmax(axis=-1).astype(np.int8)
    no_mass = ~(post > 0).any(axis=-1)
    if no_mass.any():
        codes[no_mass] = MISSING
    return codes


def iter_chunks(filemap: Dict[str, str], header: Header, chunk_rows: int = 10_000,
                keep_posteriors: bool = False, **_) -> Iterator[Chunk]:
    fn = filemap["fb.tsv"]
    S, A = header.n_samples, header.n_ancestries
    keep = keep_posteriors or header.has_posterior

    reader = pd.read_csv(
        fn, sep=r"\s+", header=None, skiprows=2, compression="infer",
        chunksize=chunk_rows, dtype={0: str},
    )
    for frame in reader:
        n = len(frame)
        chrom = frame.iloc[:, 0].astype(str).to_numpy()
        pos = frame.iloc[:, 1].to_numpy(dtype=np.int32)
        post = frame.iloc[:, _META_COLS:].to_numpy(dtype=np.float32)
        if post.shape[1] != S * 2 * A:
            raise ValueError(
                f"Row block has {post.shape[1]} data columns; expected {S * 2 * A}."
            )
        post = post.reshape(n, S, 2, A)
        codes = _codes_from_posteriors(post)
        yield Chunk(chrom, pos, pos.copy(), codes, posterior=post if keep else None)
