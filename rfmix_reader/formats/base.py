"""
Common parser interface.  Every module in :mod:`rfmix_reader.formats` exposes

* ``FORMAT``            – short name (``"msp"``, ``"fb"``, ``"flare"``, ``"haptools"``)
* ``discover(path, chrom=None)`` – list of per-chromosome file maps
* ``scan(filemap, **opts) -> Header``
* ``iter_chunks(filemap, header, chunk_rows, **opts) -> Iterator[Chunk]``

Parsers are plain numpy / pandas / cyvcf2 code: one chunk in flight, no dask,
no GPU branches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..readers._common import MISSING

__all__ = ["Header", "Chunk", "MISSING", "codes_from_pairs", "chrom_label_from_path"]


@dataclass
class Header:
    """What is known about a source before its variants are streamed."""

    samples: List[str]
    ancestries: List[str]              # tool order == axis order everywhere
    source_files: List[str]
    chrom: Optional[str] = None        # e.g. "chr21" when known up front
    n_variants: Optional[int] = None
    global_ancestry: Optional[np.ndarray] = None   # (S, A) float32 in `ancestries` order
    has_posterior: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_ancestries(self) -> int:
        return len(self.ancestries)


@dataclass
class Chunk:
    """A block of consecutive variants."""

    chrom: np.ndarray                  # (n,) str
    pos: np.ndarray                    # (n,) int32
    end: np.ndarray                    # (n,) int32 (== pos unless segments)
    codes: np.ndarray                  # (n, S, 2) int8, MISSING = -1
    posterior: Optional[np.ndarray] = None   # (n, S, 2, A) float32

    def __len__(self) -> int:
        return int(self.codes.shape[0])


def codes_from_pairs(hap0, hap1, n_anc: int) -> np.ndarray:
    """
    Stack two haplotype code arrays ``(..., )`` into ``(..., 2)`` int8 codes,
    mapping anything outside ``0..n_anc-1`` to :data:`MISSING`.
    """
    h0 = np.asarray(hap0).astype(np.int64, copy=False)
    h1 = np.asarray(hap1).astype(np.int64, copy=False)
    out = np.stack([h0, h1], axis=-1)
    bad = (out < 0) | (out >= n_anc)
    out = out.astype(np.int8)
    if bad.any():
        out[bad] = MISSING
    return out


def chrom_label_from_path(path: str) -> Optional[str]:
    """``"chr21"`` from ``.../run_chr21.fb.tsv``; ``None`` if not inferable."""
    from ..utils import _extract_chrom_from_path

    label = _extract_chrom_from_path(str(path))
    return None if label is None else f"chr{label}"
