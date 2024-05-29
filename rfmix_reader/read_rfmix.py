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

from chunk import Chunk

__all__ = ["read_rfmix"]

############ Testing ##############
file_prefix = "/dcs05/lieber/hanlab/jbenjami/projects/localQTL_manuscript/local_ancestry_rfmix/_m/*"
###################################

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
    if len(file_prefixes) == 0:
        file_prefixes = [file_prefix.replace("*", "")]

    file_prefixes = sorted(_clean_prefixes(file_prefixes))
    fn = []
    for fp in file_prefixes:
        fn.append({s: f"{fp}.{s}" for s in ["fb.tsv", "rfmix.Q"]})
        
    ## Load loci information
    pbar = tqdm(desc="Mapping loci files", total=1*len(fn), disable=not verbose)
    loci = _read_file(fn, lambda fn: _read_loci(fn["fb.tsv"]), pbar)
    pbar.close()
    if len(file_prefixes) > 1:
        if verbose:
            msg = "Multiple files read in this order: {}"
            print(msg.format([basename(f) for f in file_prefixes]))

    nmarkers = dict()
    index_offset = 0
    for i, bi in enumerate(loci):
        nmarkers[fn[i]["fb.tsv"]] = bi.shape[0]
        bi["i"] += index_offset
        index_offset += bi.shape[0]
    loci = concat(loci, axis=0, ignore_index=True)
    
    ## Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping Q files", total=1*len(fn), disable=not verbose)
    rf_q = _read_file(fn, lambda fn: _read_Q(fn["rfmix.Q"]), pbar)
    pbar.close()
    nsamples = rf_q[0].shape[0]
    rf_q = concat(rf_q, axis=0, ignore_index=True)
    
    ## Loading local ancestry by loci
    pbar = tqdm(desc="Mapping fb files", total=1*len(fn), disable=not verbose)
    admix = None
    pbar.close()
    return (loci, rf_q, admix)


def _read_file(fn, read_func, pbar):
    data = [];
    for f in fn:
        data.append(read_func(f))
        pbar.update(1)
    return data


def _read_csv(fn, header) -> DataFrame:
    df = read_csv(
        fn,
        delim_whitespace=True,
        header=None,
        names=list(header.keys()),
        dtype=header,
        comment="#",
        compression=None,
        engine="c",
        iterator=False,
    )
    assert isinstance(df, DataFrame)
    return df


def _read_tsv(fn) -> DataFrame:
    from numpy import int32
    from pandas import StringDtype
    header = {"chromosome": StringDtype(),
              "physical_position": int32}
    df = read_csv(
        fn,
        delim_whitespace=True,
        header=0,
        usecols=["chromosome", "physical_position"],
        dtype=header,
        comment="#",
        compression=None,
        engine="c",
        iterator=False,
    )
    assert isinstance(df, DataFrame)
    return df


def _read_loci(fn):
    df = _read_tsv(fn)
    df["i"] = range(df.shape[0])
    return df


def _read_Q(fn):
    from re import search
    df = _read_Q_noi(fn)
    df["chrom"] = search(r'chr(\d+)', fn).group(0)
    return df


def _read_Q_noi(fn):
    header = odict(_types(fn))
    return _read_csv(fn, header)


def _types(fn):
    from pandas import StringDtype
    df = read_csv(
        fn,
        delim_whitespace=True,
        nrows=2,
        skiprows=1,
    )    
    header = {"sample_id": StringDtype()}
    header.update(df.dtypes[1:].to_dict())
    return header


def _clean_prefixes(prefixes):
    paths = []
    for p in prefixes:
        dirn = dirname(p)
        basen = basename(p)
        base = basen.split(".")[0]
        if base != "logs":
            path = join(dirn, base)
            paths.append(path)
    return list(set(paths))
