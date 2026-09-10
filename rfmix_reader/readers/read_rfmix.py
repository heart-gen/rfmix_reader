"""
Reader for RFMix v2 ``.fb.tsv`` (forward-backward posterior) files.

Adapted from `_read.py` in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_read.py

Format (2-line header, then one row per variant)::

    #reference_panel_population:  AFR  EUR
    chromosome  physical_position  genetic_position  genetic_marker_index  S1:::hap1:::AFR  S1:::hap1:::EUR  S1:::hap2:::AFR  S1:::hap2:::EUR ...
    chr21  5030578  0.00000  0  1.00000  0.00000  1.00000  0.00000 ...

Data columns are sample-major, then haplotype, then population, so the data
block of a row reshapes to ``(n_samples, 2, n_pops)``.
"""
from __future__ import annotations

import gzip
from os.path import basename, join, exists
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from dask.array import Array, concatenate, map_blocks

from .fb_read import read_fb
from ._common import (
    align_g_anc_columns,
    check_pop_order_consistent,
    counts_from_hap_codes,
    maybe_report_gpu,
    maybe_to_backend_frames,
    read_rfmix_q,
)
from ..io.chunk import Chunk
from ..io.errors import BinaryFileNotFoundError
from ..utils import (
    _read_file,
    create_binaries,
    filter_file_maps_by_chrom,
    get_prefixes,
)

__all__ = ["read_rfmix_fb"]


def read_rfmix_fb(
        file_prefix: str, binary_dir: str = "./binary_files",
        generate_binary: bool = False, verbose: bool = True,
        return_original: bool = False,
        chrom: Optional[str] = None,
        chunk: Optional[Chunk] = None,
) -> (
    Tuple[pd.DataFrame, pd.DataFrame, Array]
    | Tuple[pd.DataFrame, pd.DataFrame, Array, Array]
):
    """
    Read RFMix ``.fb.tsv`` files (forward-backward posteriors) into DataFrames and a Dask array.

    Use this reader when your analysis requires the **posterior probability
    values** (``return_original=True``).  For hard ancestry calls the much
    smaller ``.msp.tsv`` files read by :func:`read_rfmix` are sufficient.

    Parameters
    ----------
    file_prefix : str
        Directory, or path prefix, of the RFMix output files.  All chromosomes
        found are loaded unless ``chrom`` is given.
    binary_dir : str, optional
        Directory holding the binary versions of the ``.fb.tsv`` files
        (see :func:`create_binaries`).  Default ``"./binary_files"``.
    generate_binary : bool, optional
        Generate the binary files before reading.  Default ``False``.
    verbose : bool, optional
        Show progress bars.  Default ``True``.
    return_original : bool, optional
        Also return ``X_raw``, the raw float32 posterior matrix.
    chrom : str, optional
        Restrict reading to one chromosome (with or without ``chr`` prefix).
    chunk : Chunk, optional
        Dask block sizes for the posterior matrix.  Default ``Chunk()``.

    Returns
    -------
    loci_df : pandas.DataFrame
        ``chromosome``, ``physical_position``, ``i``; one row per variant.
    g_anc : pandas.DataFrame
        Global ancestry per chromosome from ``.rfmix.Q``.  Ancestry columns
        are in the RFMix reference-panel order, which is also the order of
        axis 2 of ``local_array``.
    local_array : dask.array.Array, int8
        Shape ``(variants, samples, ancestries)``.  Hard diploid ancestry
        counts ``0/1/2`` obtained by taking, for each haplotype, the ancestry
        with the highest posterior.  ``-1`` marks a haplotype with no
        posterior mass (all zeros).
    X_raw : dask.array.Array, float32, optional
        Only when ``return_original`` is true.  Shape
        ``(variants, samples * 2 * ancestries)``, columns in file order.

    Notes
    -----
    Populations of the returned arrays follow the ``#reference_panel_population``
    header of the ``.fb.tsv`` file so that ``get_pops(g_anc)`` labels axis 2.
    """
    chunk = chunk or Chunk()
    maybe_report_gpu(verbose)

    fn = filter_file_maps_by_chrom(
        get_prefixes(file_prefix, "rfmix", verbose), chrom, kind="RFMix"
    )

    # Population order from the .fb.tsv header (authoritative for axis 2)
    pops = check_pop_order_consistent(
        [_read_fb_pops(f["fb.tsv"]) for f in fn], kind="RFMix .fb.tsv"
    )

    # Loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn), disable=not verbose)
    loci_dfs = _read_file(fn, lambda f: _read_loci(f["fb.tsv"]), pbar)
    pbar.close()

    nmarkers = {}
    index_offset = 0
    for f, df in zip(fn, loci_dfs):
        nmarkers[f["fb.tsv"]] = df.shape[0]
        df["i"] += index_offset
        index_offset += df.shape[0]
    loci_df = pd.concat(loci_dfs, axis=0, ignore_index=True)

    # Global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn),
                disable=not verbose)
    g_anc_list = _read_file(fn, lambda f: _read_Q(f["rfmix.Q"]), pbar)
    pbar.close()

    nsamples = g_anc_list[0].shape[0]
    g_anc = pd.concat(g_anc_list, axis=0, ignore_index=True)
    g_anc = align_g_anc_columns(g_anc, pops)

    # Local ancestry
    if generate_binary:
        create_binaries(file_prefix, binary_dir, chrom=chrom, verbose=verbose)

    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn),
                disable=not verbose)
    local_data = _read_file(
        fn,
        lambda f: _read_fb(
            f["fb.tsv"], nsamples, nmarkers[f["fb.tsv"]], pops, binary_dir, chunk,
        ),
        pbar,
    )
    pbar.close()

    local_array = concatenate([admix for admix, _ in local_data], axis=0)
    loci_df, g_anc = maybe_to_backend_frames(loci_df, g_anc)

    if return_original:
        X_raw = concatenate([X for _, X in local_data], axis=0)
        return loci_df, g_anc, local_array, X_raw
    return loci_df, g_anc, local_array


def _read_fb_pops(fn: str) -> List[str]:
    """Population labels, in file order, from the first line of a ``.fb.tsv``."""
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt") as fh:
        line = fh.readline().strip()
    if not line.startswith("#reference_panel_population"):
        raise ValueError(
            f"Unexpected first line in '{fn}'. Expected "
            "'#reference_panel_population:\\tPOP1\\tPOP2 ...'."
        )
    pops = line.split(":", 1)[1].split()
    if not pops:
        raise ValueError(f"No populations listed in the header of '{fn}'.")
    return pops


def _read_tsv(fn: str) -> pd.DataFrame:
    """Read ``chromosome`` and ``physical_position`` from a ``.fb.tsv`` file."""
    header = {"chromosome": "category", "physical_position": np.int32}
    try:
        chunks = pd.read_csv(
            fn, sep=r"\s+", header=0, usecols=list(header.keys()),
            dtype=header, comment="#", compression="infer",
            chunksize=100_000,  # low-memory chunks
        )
        df = pd.concat(chunks, ignore_index=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fn} not found.")
    except Exception as e:
        raise OSError(f"Error reading file {fn}: {e}") from e

    if not all(column in df.columns for column in header):
        raise ValueError(f"DataFrame does not contain expected columns: {list(header)}")
    return df


def _read_loci(fn: str) -> pd.DataFrame:
    """Loci table with a sequential ``i`` index column."""
    df = _read_tsv(fn)
    df["i"] = np.arange(df.shape[0], dtype=np.int64)
    return df


def _read_Q(fn: str) -> pd.DataFrame:
    """Q matrix with a ``chrom`` column inferred from the file name."""
    return read_rfmix_q(fn, add_chrom=True)


def _read_Q_noi(fn: str) -> pd.DataFrame:
    """Q matrix without the ``chrom`` column."""
    return read_rfmix_q(fn, add_chrom=False)


def _read_fb(
    fn: str, nsamples: int, nloci: int, pops: list, temp_dir: str,
    chunk: Optional[Chunk] = None,
) -> Tuple[Array, Array]:
    """
    Read the binary forward-backward matrix as lazy dask arrays.

    Returns
    -------
    admix : dask.array.Array, int8, ``(nloci, nsamples, npops)``
        Hard diploid counts (see :func:`_posteriors_to_counts`).
    X : dask.array.Array, float32, ``(nloci, nsamples * 2 * npops)``
        Raw posteriors.
    """
    chunk = chunk or Chunk()
    npops = len(pops)
    stride = 2 * npops  # columns per sample
    nrows = nloci
    ncols = nsamples * stride

    row_chunk = nrows if chunk.nloci is None else min(nrows, chunk.nloci)
    col_chunk = ncols if chunk.nsamples is None else min(ncols, chunk.nsamples * stride)
    max_npartitions = 16_384
    row_chunk = max(nrows // max_npartitions, row_chunk)
    col_chunk = max(ncols // max_npartitions, col_chunk)
    # Column blocks must hold whole samples.
    col_chunk = max(stride, (col_chunk // stride) * stride)

    binary_fn = join(temp_dir, basename(fn).split(".")[0] + ".bin")
    if not exists(binary_fn):
        raise BinaryFileNotFoundError(binary_fn, temp_dir)

    X = read_fb(binary_fn, nrows, ncols, row_chunk, col_chunk)
    admix = _posteriors_to_counts(X, npops)
    return admix, X


def _block_counts(block: np.ndarray, npops: int) -> np.ndarray:
    """Per-block posterior → hard diploid counts (numpy)."""
    nloci, ncols = block.shape
    nsamples = ncols // (2 * npops)
    b4 = block.reshape(nloci, nsamples, 2, npops)
    codes = b4.argmax(axis=-1)
    no_mass = ~(b4 > 0).any(axis=-1)  # haplotype with all-zero posteriors
    codes = np.where(no_mass, -1, codes)
    return counts_from_hap_codes(codes[..., 0], codes[..., 1], npops)


def _posteriors_to_counts(X: Array, npops: int) -> Array:
    """
    Convert the raw posterior matrix into hard diploid ancestry counts.

    For every sample and haplotype the ancestry with the highest posterior is
    taken; the two haplotype calls are then summed into ``0/1/2`` counts per
    ancestry (int8).  A haplotype whose posteriors are all zero yields
    ``-1`` for every ancestry of that sample/locus.

    Parameters
    ----------
    X : dask.array.Array, ``(nloci, nsamples * 2 * npops)``
    npops : int

    Returns
    -------
    dask.array.Array, int8, ``(nloci, nsamples, npops)``
    """
    ncols = int(X.shape[1])
    stride = 2 * npops
    if ncols % stride != 0:
        raise ValueError(
            "The number of columns in X must be divisible by (2 * npops). "
            "Expected layout: 2 haplotypes per sample per ancestry."
        )
    if any(c % stride for c in X.chunks[1]):
        # Make every column block hold whole samples.
        X = X.rechunk({1: ncols})

    out_chunks = (X.chunks[0], tuple(c // stride for c in X.chunks[1]), (npops,))
    return map_blocks(
        _block_counts, X, npops, dtype=np.int8, new_axis=2, chunks=out_chunks,
    )
