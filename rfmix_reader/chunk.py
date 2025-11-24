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
    Chunk specification for a contiguous submatrix of the haplotype matrix.

    Parameters
    ----------
    nsamples : Optional[int], default=1024
        Number of samples in a single chunk, limited by the total number of
        samples. Set to `None` to include all samples.
    nloci : Optional[int], default=1024
        Number of loci in a single chunk, limited by the total number of
        loci. Set to `None` to include all loci.

    Notes
    -----
    - Small chunks may increase computational time, while large chunks may increase
      memory usage.
    - For small datasets, try setting both `nsamples` and `nloci` to `None`.
    - For large datasets where you need to use every sample, try setting `nsamples=None`
      and choose a small value for `nloci`.
    """
    nsamples: Optional[int] = 1024
    nloci: Optional[int] = 1024
