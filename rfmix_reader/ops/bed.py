"""Constant-ancestry intervals (BED-like) for one sample."""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..core import schema as S

__all__ = ["to_bed", "resolve_sample"]


def resolve_sample(ds: xr.Dataset, sample: Union[int, str]) -> int:
    """Sample index from an integer position or a sample ID."""
    samples = ds.la.samples
    if isinstance(sample, (int, np.integer)):
        idx = int(sample)
        if idx < 0 or idx >= len(samples):
            raise IndexError(f"sample {idx} is out of range [0, {len(samples) - 1}]")
        return idx
    try:
        return samples.index(str(sample))
    except ValueError:
        raise KeyError(f"sample {sample!r} not found") from None


def run_boundaries(states: np.ndarray, min_segment: int = 1) -> List[Tuple[int, int, int]]:
    """
    Runs of identical rows in ``states`` ``(n, k)``.

    Returns ``(start, stop_exclusive, representative_row)`` triples.  Runs
    shorter than ``min_segment`` are absorbed into the preceding run (or the
    following one at the start), and adjacent runs with equal state are merged.
    """
    n = states.shape[0]
    if n == 0:
        return []
    change = np.flatnonzero((states[1:] != states[:-1]).any(axis=1)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [n]])
    runs = [[int(s), int(e), int(s)] for s, e in zip(starts, stops)]

    if min_segment > 1:
        merged: List[List[int]] = []
        for run in runs:
            if run[1] - run[0] < min_segment and merged:
                merged[-1][1] = run[1]              # absorb into previous run
            elif run[1] - run[0] < min_segment and not merged:
                merged.append(run)                   # provisional first run
            elif merged and merged[-1][1] - merged[-1][0] < min_segment:
                merged[-1] = [merged[-1][0], run[1], run[2]]  # first run was short: absorb it
            else:
                merged.append(run)
        runs = merged
        # merge neighbours that ended up with the same state
        out: List[List[int]] = []
        for run in runs:
            if out and np.array_equal(states[out[-1][2]], states[run[2]]):
                out[-1][1] = run[1]
            else:
                out.append(run)
        runs = out
    return [tuple(r) for r in runs]


def to_bed(ds: xr.Dataset, sample: Union[int, str], *, min_segment: int = 1) -> pd.DataFrame:
    """
    BED-like intervals of constant diploid ancestry for ``sample``.

    Columns: ``chromosome``, ``start`` (position of the first variant of the
    run), ``end`` (segment end of the last variant), then one
    ``<sample>_<ancestry>`` count column per ancestry.

    Parameters
    ----------
    sample : int or str
        Sample index or ID.
    min_segment : int
        Runs shorter than this many variants are absorbed into their
        neighbour (noise suppression).
    """
    idx = resolve_sample(ds, sample)
    name = ds.la.samples[idx]
    pops = ds.la.ancestries
    counts = np.asarray(ds.la.counts.isel({S.SAMPLE: idx}).values)      # (L, A)
    chrom = np.asarray(ds[S.CHROMOSOME].values).astype(str)
    pos = np.asarray(ds[S.VARIANT_POSITION].values)
    end = np.asarray(ds[S.SEGMENT_END].values) if S.SEGMENT_END in ds.coords else pos

    rows = []
    for c in dict.fromkeys(chrom.tolist()):
        where = np.flatnonzero(chrom == c)
        order = where[np.argsort(pos[where], kind="stable")]
        sub = counts[order]
        for s, e, rep in run_boundaries(sub, min_segment):
            rows.append((c, int(pos[order[s]]), int(end[order[e - 1]]), *sub[rep].tolist()))

    cols = ["chromosome", "start", "end"] + [f"{name}_{p}" for p in pops]
    bed = pd.DataFrame(rows, columns=cols)
    for col in cols[3:]:
        bed[col] = bed[col].astype(np.int8)
    return bed
