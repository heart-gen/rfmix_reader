"""
Adapted from the `_chunk.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_chunk.py
"""
from typing import Optional
from dataclasses import dataclass

__all__ = ["Chunk"]


@dataclass
class Chunk:
    """
    Chunk specification for reading the RFMix forward-backward binary.

    Parameters
    ----------
    nsamples : Optional[int], default=1024
        Number of **samples** per dask block along the column axis.  Each
        sample occupies ``2 * n_ancestries`` columns, so blocks always hold
        whole samples.  ``None`` puts every sample in one block.
    nloci : Optional[int], default=1024
        Number of loci per dask block along the row axis.  ``None`` puts every
        locus in one block.

    Notes
    -----
    - Small chunks increase scheduling overhead; large chunks increase memory.
    - For small datasets, set both to ``None``.
    - For large datasets that need every sample, set ``nsamples=None`` and
      choose a moderate ``nloci``.
    """
    nsamples: Optional[int] = 1024
    nloci: Optional[int] = 1024
