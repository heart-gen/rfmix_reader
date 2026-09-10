"""Query local ancestry at arbitrary genomic positions."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ..core import schema as S
from ..formats.common import normalize_chrom_label

__all__ = ["at_positions"]


def _locate(starts: np.ndarray, ends: np.ndarray, query: np.ndarray, method: str):
    """Index of the source variant for each query position (``-1`` = none)."""
    if starts.size == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    k = np.searchsorted(starts, query, side="right") - 1
    if method == "stepwise":
        safe = np.clip(k, 0, starts.size - 1)
        ok = (k >= 0) & (query <= ends[safe])
        return np.where(ok, k, -1)
    if method == "nearest":
        left = np.clip(k, 0, starts.size - 1)
        right = np.clip(k + 1, 0, starts.size - 1)
        d_left = np.abs(query - starts[left])
        d_right = np.abs(starts[right] - query)
        return np.where(d_left <= d_right, left, right)
    raise ValueError("method must be 'stepwise' or 'nearest'.")


def at_positions(
    ds: xr.Dataset, loci: pd.DataFrame, *, chrom_col: str = "chrom", pos_col: str = "pos",
    samples: Optional[Sequence[str]] = None, aggregate: bool = True,
    method: str = "stepwise",
) -> pd.DataFrame:
    """
    Local ancestry at the positions listed in ``loci``.

    Parameters
    ----------
    loci : DataFrame
        Must contain ``chrom_col`` and ``pos_col``; every other column is
        carried through to the output.
    samples : sequence of str, optional
        Restrict to these sample IDs (default: all).
    aggregate : bool
        ``True``: one row per locus with ``n_samples``, ``n_haplotypes`` and
        ``<pop>_haplotypes`` / ``<pop>_fraction`` columns.
        ``False``: one row per locus and sample with ``sample_id`` and
        ``<pop>_copies`` columns.
    method : {"stepwise", "nearest"}
        ``stepwise`` matches a position to the segment/variant whose
        ``[variant_position, segment_end]`` interval contains it (closed on
        both sides); ``nearest`` takes the closest variant.

    Returns
    -------
    DataFrame
        In the order of ``loci``; ``matched`` says whether a source variant
        was found (unmatched rows carry NaN ancestry values).
    """
    if chrom_col not in loci.columns or pos_col not in loci.columns:
        raise ValueError(f"loci must contain '{chrom_col}' and '{pos_col}' columns.")
    if loci.empty:
        return loci.copy()

    pops = ds.la.ancestries
    all_samples = ds.la.samples
    if samples is None:
        sample_idx = np.arange(len(all_samples))
        selected = list(all_samples)
    else:
        missing = [s for s in samples if s not in all_samples]
        if missing:
            raise ValueError(f"Samples not found: {missing}")
        sample_idx = np.array([all_samples.index(s) for s in samples])
        selected = [str(s) for s in samples]

    q_chrom = np.array([normalize_chrom_label(str(v).strip()) for v in loci[chrom_col]])
    q_pos = pd.to_numeric(loci[pos_col], errors="raise").to_numpy(dtype=np.int64)
    src_chrom = np.array([normalize_chrom_label(str(c)) for c in ds[S.CHROMOSOME].values])
    src_pos = np.asarray(ds[S.VARIANT_POSITION].values, dtype=np.int64)
    src_end = (np.asarray(ds[S.SEGMENT_END].values, dtype=np.int64)
               if S.SEGMENT_END in ds.coords else src_pos)

    hit = np.full(len(loci), -1, dtype=np.int64)   # global variant index per query
    for c in np.unique(q_chrom):
        q = np.flatnonzero(q_chrom == c)
        v = np.flatnonzero(src_chrom == c)
        if v.size == 0:
            continue
        order = v[np.argsort(src_pos[v], kind="stable")]
        k = _locate(src_pos[order], src_end[order], q_pos[q], method)
        hit[q] = np.where(k >= 0, order[np.clip(k, 0, None)], -1)

    matched = hit >= 0
    uniq, inverse = np.unique(hit[matched], return_inverse=True)
    if uniq.size:
        gathered = np.asarray(ds.la.counts.isel({S.VARIANT: uniq, S.SAMPLE: sample_idx}).values)
    else:
        gathered = np.zeros((0, len(selected), len(pops)), dtype=np.int8)
    counts = np.full((len(loci), len(selected), len(pops)), -1, dtype=np.int16)
    counts[matched] = gathered[inverse]

    base = loci.reset_index(drop=True).copy()
    base["matched"] = matched
    valid = (counts >= 0).all(axis=2)                      # (n, S)
    if aggregate:
        per_pop = np.where(valid[..., None], counts, 0).sum(axis=1).astype(float)   # (n, A)
        total = per_pop.sum(axis=1)
        base["n_samples"] = len(selected)
        base["n_haplotypes"] = total.astype(int)
        for a, pop in enumerate(pops):
            hap = per_pop[:, a]
            base[f"{pop}_haplotypes"] = np.where(matched, hap, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                base[f"{pop}_fraction"] = np.where(matched & (total > 0), hap / total, np.nan)
        return base

    n = len(loci)
    out = base.loc[np.repeat(np.arange(n), len(selected))].reset_index(drop=True)
    out["sample_id"] = np.tile(np.asarray(selected, dtype=object), n)
    flat_valid = valid.reshape(-1)
    for a, pop in enumerate(pops):
        vals = counts[:, :, a].reshape(-1).astype(float)
        out[f"{pop}_copies"] = np.where(flat_valid, vals, np.nan)
    return out
