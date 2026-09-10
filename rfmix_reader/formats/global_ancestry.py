"""Global-ancestry tables (RFMix ``.rfmix.Q``, FLARE ``.global.anc.gz``) as arrays."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..readers._common import read_rfmix_q

__all__ = ["read_rfmix_q", "read_flare_global", "frame_to_array", "fractions_from_counts"]


def read_flare_global(fn: str) -> pd.DataFrame:
    """FLARE ``.global.anc.gz`` as ``sample_id`` + one float32 column per ancestry."""
    from ..readers.read_flare import _read_anc_noi

    return _read_anc_noi(fn)


def frame_to_array(df: Optional[pd.DataFrame], samples: Sequence[str],
                   ancestries: Sequence[str]) -> Optional[np.ndarray]:
    """
    ``(S, A)`` float32 array of ``df`` aligned to ``samples`` / ``ancestries``.

    Returns ``None`` when ``df`` is None.  Raises if a sample or ancestry is
    missing from the table.
    """
    if df is None:
        return None
    missing_anc = [a for a in ancestries if a not in df.columns]
    if missing_anc:
        raise ValueError(
            f"Global ancestry table lacks columns {missing_anc}; "
            f"expected {list(ancestries)} (found {list(df.columns)})."
        )
    table = df.set_index(df["sample_id"].astype(str))
    missing_samples = [s for s in samples if s not in table.index]
    if missing_samples:
        raise ValueError(
            f"Global ancestry table lacks {len(missing_samples)} sample(s), "
            f"e.g. {missing_samples[:3]}."
        )
    return table.loc[list(samples), list(ancestries)].to_numpy(dtype=np.float32)


def fractions_from_counts(counts: np.ndarray) -> np.ndarray:
    """``(S, A)`` haplotype counts -> per-sample fractions (0 where a sample has none)."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(totals > 0, counts / totals, 0.0)
    return frac.astype(np.float32)
