"""TAGORE-ready BED annotation for one sample."""
from __future__ import annotations

from typing import Union

import pandas as pd
import xarray as xr

from .bed import to_bed

__all__ = ["to_tagore"]


def to_tagore(ds: xr.Dataset, sample: Union[int, str], *, palette: str = "tab10",
              min_segment: int = 1) -> pd.DataFrame:
    """
    BED table annotated for TAGORE (``#chr``, ``start``, ``stop``, ``feature``,
    ``size``, ``color``, ``chrCopy``).  Requires the ``viz`` extra.
    """
    from ..viz.visualization import _annotate_tagore

    bed = to_bed(ds, sample, min_segment=min_segment)
    sample_cols = list(bed.columns[3:])
    return _annotate_tagore(bed, sample_cols, ds.la.ancestries, palette)
