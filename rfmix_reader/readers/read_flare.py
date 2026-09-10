"""
Reader for FLARE local-ancestry output (``.anc.vcf.gz`` + ``.global.anc.gz``).

The VCF carries per-haplotype ancestry codes in the ``AN1`` / ``AN2`` FORMAT
fields; the ``##ANCESTRY=<EUR=0,AFR=1>`` header line maps codes to labels.
Axis 2 of the returned ancestry array follows those codes.
"""
from __future__ import annotations

import warnings
from os.path import exists
from re import search
from typing import Dict, Tuple, Iterator, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from cyvcf2 import VCF
from dask import delayed
from dask.array import Array, concatenate, from_delayed

from ._common import (
    align_g_anc_columns,
    check_pop_order_consistent,
    counts_from_hap_codes,
    maybe_report_gpu,
    maybe_to_backend_frames,
    pops_by_code,
)
from ..utils import (
    _extract_chrom_from_path,
    _read_file,
    filter_file_maps_by_chrom,
    get_prefixes,
)

__all__ = ["read_flare"]


def read_flare(
        file_prefix: str, chunk_size: int = 1_000_000, verbose: bool = True,
        chrom: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Array]:
    """
    Read FLARE files into data frames and a Dask array.

    Parameters
    ----------
    file_prefix : str
        Directory, file, or path prefix of the FLARE output files.  All
        chromosomes found are loaded unless ``chrom`` is given.
    chunk_size : int
        Number of records per chunk when reading loci; ``chunk_size / 100``
        records per dask block for the ancestry array.
    verbose : bool, optional
        Show progress bars.  Default ``True``.
    chrom : str, optional
        Restrict parsing to a single chromosome (matching with or without a
        ``chr`` prefix).

    Returns
    -------
    loci_df : DataFrame
        ``chromosome``, ``physical_position``, ``i``; one row per variant.
    g_anc : DataFrame
        Global ancestry by chromosome from ``.global.anc.gz``, ancestry
        columns ordered like axis 2 of ``local_array``.
    local_array : dask.array.Array, int8
        Shape ``(variants, samples, ancestries)`` with diploid counts
        ``0/1/2``; ``-1`` where a haplotype ancestry is missing.  Axis 2
        follows the codes of the ``##ANCESTRY`` header.
    """
    maybe_report_gpu(verbose)

    fn = filter_file_maps_by_chrom(
        get_prefixes(file_prefix, "flare", verbose), chrom, kind="FLARE"
    )

    pops = check_pop_order_consistent(
        [pops_by_code(_parse_ancestry_header(f["anc.vcf"])) for f in fn],
        kind="FLARE",
    )

    # Loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn),
                disable=not verbose)
    loci_dfs = _read_file(fn, lambda f: _read_loci(f["anc.vcf"], chunk_size), pbar)
    pbar.close()

    index_offset = 0
    for df in loci_dfs:  # modify in-place
        df["i"] = np.arange(index_offset, index_offset + df.shape[0], dtype=np.int64)
        index_offset += df.shape[0]
    loci_df = pd.concat(loci_dfs, axis=0, ignore_index=True)

    # Global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn),
                disable=not verbose)
    g_anc_list = _read_file(fn, lambda f: _read_anc(f["global.anc"]), pbar)
    pbar.close()
    g_anc = pd.concat(g_anc_list, axis=0, ignore_index=True)
    g_anc = align_g_anc_columns(g_anc, pops)

    # Local ancestry
    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn),
                disable=not verbose)
    local_arrays = _read_file(
        fn,
        lambda f: _load_haplotypes(f["anc.vcf"], max(1, int(chunk_size / 100))),
        pbar,
    )
    pbar.close()
    local_array = concatenate(local_arrays, axis=0)

    loci_df, g_anc = maybe_to_backend_frames(loci_df, g_anc)
    return loci_df, g_anc, local_array


def _read_vcf(fn: str, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Read ``chromosome`` / ``physical_position`` of every record into a DataFrame."""
    try:
        chunks = list(_load_vcf_info(fn, chunk_size))
        df = pd.concat(chunks, ignore_index=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fn} not found.")
    except Exception as e:
        raise OSError(f"Error reading file {fn}: {e}") from e

    df["chromosome"] = df["chromosome"].astype("category")
    df["physical_position"] = df["physical_position"].astype(np.int32)
    return df


def _read_loci(fn: str, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """Loci table with a sequential ``i`` index column."""
    df = _read_vcf(fn, chunk_size)
    df["i"] = np.arange(df.shape[0], dtype=np.int64)
    return df


def _read_anc_noi(fn: str) -> pd.DataFrame:
    """
    Read a FLARE ``.global.anc.gz`` table without the ``chrom`` column.

    Format (tab separated, one header line)::

        SAMPLE  EUR  AFR
        Sample_1  0.7  0.3
    """
    try:
        df = pd.read_csv(fn, sep="\t", compression="infer")
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{fn}' not found.")
    except Exception as e:
        raise OSError(f"Error reading file {fn}: {e}") from e
    if df.shape[1] < 2:
        raise ValueError(f"Global ancestry file '{fn}' has no ancestry columns.")

    df = df.rename(columns={df.columns[0]: "sample_id"})
    df["sample_id"] = df["sample_id"].astype(str)
    for col in df.columns[1:]:
        df[col] = df[col].astype(np.float32)
    return df


def _read_anc(fn: str) -> pd.DataFrame:
    """Global ancestry table with a ``chrom`` column inferred from the file name."""
    df = _read_anc_noi(fn)
    label = _extract_chrom_from_path(fn)
    if label is not None:
        df["chrom"] = f"chr{label}"
    else:
        warnings.warn(
            f"Could not extract chromosome information from '{fn}'", stacklevel=2
        )
    return df


def _load_haplotypes(vcf_file: str, chunk_size: int = 10_000) -> Array:
    """
    Load diploid ancestry counts from a FLARE VCF into a dask array.

    The ``##ANCESTRY`` header gives the code → label mapping; axis 2 of the
    result is ordered by code.  Each sample/variant sums the ancestries of the
    two haplotypes (``AN1``, ``AN2``) into ``0/1/2`` counts (int8).  A missing
    or out-of-range code yields ``-1`` for every ancestry of that sample.

    Parameters
    ----------
    vcf_file : str
        Path to the FLARE VCF (``.anc.vcf`` or ``.anc.vcf.gz``).
    chunk_size : int, optional
        Number of variant records per dask block.  Default 10,000.

    Returns
    -------
    dask.array.Array, int8, shape ``(num_variants, num_samples, num_ancestries)``
    """
    if not exists(vcf_file):
        raise FileNotFoundError(f"VCF file not found: {vcf_file}")

    ancestry_map = _parse_ancestry_header(vcf_file)
    n_ancestries = len(pops_by_code(ancestry_map))

    vcf = VCF(vcf_file)
    try:
        n_samples = len(vcf.samples)

        def process_chunk(pairs):
            """pairs: list of (an1, an2) int32 arrays of shape (n_samples,)."""
            an1 = np.stack([p[0] for p in pairs], axis=0)
            an2 = np.stack([p[1] for p in pairs], axis=0)
            return counts_from_hap_codes(an1, an2, n_ancestries)

        records_buffer = []  # holds (an1, an2) numpy pairs, not Variant objects
        delayed_arrays = []
        for rec in vcf:
            an1 = np.asarray(rec.format("AN1"), dtype=np.int32).ravel()
            an2 = np.asarray(rec.format("AN2"), dtype=np.int32).ravel()
            records_buffer.append((an1, an2))
            if len(records_buffer) == chunk_size:
                delayed_arrays.append(
                    from_delayed(
                        delayed(process_chunk)(records_buffer),
                        shape=(chunk_size, n_samples, n_ancestries),
                        dtype=np.int8,
                    )
                )
                records_buffer = []

        if records_buffer:
            delayed_arrays.append(
                from_delayed(
                    delayed(process_chunk)(records_buffer),
                    shape=(len(records_buffer), n_samples, n_ancestries),
                    dtype=np.int8,
                )
            )
    finally:
        vcf.close()

    if not delayed_arrays:
        raise ValueError(f"No variant records found in {vcf_file}")
    return concatenate(delayed_arrays, axis=0)


def _diploid_counts_from_haps(an1, an2, n_ancestries: int) -> np.ndarray:
    """Diploid counts (int8, ``-1`` for missing) from two haplotype code arrays."""
    return counts_from_hap_codes(an1, an2, n_ancestries)


def _parse_ancestry_header(vcf_file: str) -> Dict[str, int]:
    """
    Parse ancestry population index from the VCF header.

    Looks for a line starting with '##ANCESTRY=' formatted like:
    '##ANCESTRY=<EUR=0,AFR=1>'

    Returns
    -------
    dict
        Mapping from ancestry label (e.g., 'EUR') to integer index (e.g., 0).
    """
    vcf = VCF(vcf_file)
    try:
        ancestries: Dict[str, int] = {}
        for hline in vcf.raw_header.splitlines():
            if hline.startswith("##ANCESTRY="):
                m = search(r"<(.+)>", hline)
                if m:
                    for pair in m.group(1).split(","):
                        label, idx = pair.split("=")
                        ancestries[label.strip()] = int(idx)
                break
    finally:
        vcf.close()
    if not ancestries:
        raise ValueError(f"No '##ANCESTRY=<...>' header line found in {vcf_file}")
    return ancestries


def _load_vcf_info(vcf_file: str, chunk_size: int = 1_000_000
                   ) -> Iterator[pd.DataFrame]:
    """
    Yield DataFrames of ``chromosome`` / ``physical_position`` in chunks.
    """
    vcf = VCF(vcf_file)
    try:
        chroms, positions = [], []
        for rec in vcf:
            chroms.append(rec.CHROM)
            positions.append(rec.POS)
            if len(chroms) == chunk_size:
                yield pd.DataFrame({"chromosome": chroms, "physical_position": positions})
                chroms, positions = [], []
        if chroms:
            yield pd.DataFrame({"chromosome": chroms, "physical_position": positions})
    finally:
        vcf.close()
