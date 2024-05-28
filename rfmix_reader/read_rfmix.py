"""
Adapted from `_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py
"""
import warnings
from glob import glob
from typing import Optional
from os.path import basename, dirname, join
from collections import OrderedDict as odict

from xarray import DataArray
from pandas import DataFrame, read_csv

from .chunk import Chunk

__all__ = ["read_rfmix"]


def read_rfmix(file_prefix, verbose=True):
    """
    Read RFMix files into data frames.

    Notes
    -----
    Local ancestry can be either :const:`0`, :const:`1`, :const:`2`, or
    :data:`math.nan`:

    - :const:`0` No alleles are associated with this ancestry
    - :const:`1` One allele is associated with this ancestry
    - :const:`2` Both alleles are associated with this ancestry

    Parameters
    ----------
    file_prefix : str
        Path prefix to the set of RFMix files. It will load all of the chromosomes
        at once.p
    verbose : bool
        :const:`True` for progress information; :const:`False` otherwise.

    Returns
    -------
    xxx
    xxx
    xxx
    """
    from tqdm import tqdm
    from pandas import concat
    from dask.array import concatenate

    file_prefixes = sorted(glob(file_prefix))
    if len(file_prefix) == 0:
        file_prefixes = [file_prefix.replace("*", "")]

    file_prefixes = sorted(_clean_prefixes(file_prefixes))
    return None


def _clean_prefixes(prefixes):
    paths = []
    for p in prefixes:
        dirn = dirname(p)
        basen = basename(p)
        base = ".".join(basen.split(".")[:-1])
        if len(base) == 0:
            path = p
        else:
            ext = basen.split(".")[-1]
            if ext not in ["Q", "tsv"]:
                base += "." + ext
            path = join(dirn, base)
        paths.append(path)
    return list(set(paths))
