"""
Reader for RFMix v2 `.msp.tsv` (Most-probable Sequence of Populations) files.

The `.msp.tsv` format stores piecewise-constant hard ancestry calls as genomic
segments — roughly 2,000× smaller than the `.fb.tsv` forward-backward matrix
for the same dataset.  Reading `.msp.tsv` eliminates the need for binary
conversion and is the preferred source when posterior probabilities are not
required (e.g. GWAS, admixture mapping, local-ancestry validation).

Format (2-line header, then one row per segment):
    #Subpopulation order/codes: AFR=0   EUR=1
    #chm  spos  epos  sgpos  egpos  n snps  Sample_1.0  Sample_1.1 ...
    chr1  10397 670270 0.00  0.10   4752    0            1          ...

The per-sample columns alternate haplotype 0 / haplotype 1 (suffix .0 / .1).
Values are integer ancestry codes (0 = first listed population, 1 = second, …).
"""
from __future__ import annotations

import gzip
import re
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Sequence

from ._common import (
    align_g_anc_columns,
    check_pop_order_consistent,
    counts_from_hap_codes,
    maybe_report_gpu,
    maybe_to_backend_frames,
    pops_by_code,
    read_rfmix_q,
)
from ..utils import (
    _normalize_chrom_label,
    _read_file,
    filter_file_maps_by_chrom,
    get_prefixes,
)

__all__ = ["read_rfmix", "extract_locus_ancestry"]

_META_COLS = ["#chm", "spos", "epos", "sgpos", "egpos", "n snps"]


def _parse_pop_header(fn: str) -> Dict[str, int]:
    """
    Parse the first header line of an .msp.tsv file to extract population codes.

    Expected format:
        #Subpopulation order/codes: AFR=0   EUR=1

    Returns a dict mapping population label → integer code, e.g. {"AFR": 0, "EUR": 1}.
    """
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt") as fh:
        line = fh.readline().strip()
    pairs = re.findall(r"(\w+)=(\d+)", line)
    if not pairs:
        raise ValueError(
            f"Could not parse population codes from first line of '{fn}'. "
            f"Expected format: #Subpopulation order/codes: AFR=0 EUR=1 ..."
        )
    return {label: int(code) for label, code in pairs}


def _read_msp_file(fn: str) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Read a single .msp.tsv file.

    Returns
    -------
    segs : DataFrame
        All segment rows with columns: chrom, spos, epos, and per-haplotype
        ancestry integer columns (Sample_1.0, Sample_1.1, …).
    hap_cols : list of str
        Names of the per-haplotype columns in the order they appear.
    pop_map : dict
        Population label → integer code from the file header.
    """
    pop_map = _parse_pop_header(fn)

    segs = pd.read_csv(
        fn, sep="\t", comment=None, skiprows=1, header=0,
        compression="infer",
    )
    segs.rename(columns={"#chm": "chrom"}, inplace=True)
    segs["chrom"] = segs["chrom"].astype("category")
    segs["spos"] = segs["spos"].astype(np.int32)
    segs["epos"] = segs["epos"].astype(np.int32)

    hap_cols = [c for c in segs.columns if c not in
                {"chrom", "spos", "epos", "sgpos", "egpos", "n snps"}]
    return segs, hap_cols, pop_map


def _sample_names_from_hap_cols(hap_cols: Sequence[str]) -> List[str]:
    if len(hap_cols) % 2 != 0:
        raise ValueError(
            f"Expected an even number of haplotype columns, got {len(hap_cols)}."
        )
    samples = []
    for i in range(0, len(hap_cols), 2):
        h0 = hap_cols[i]
        h1 = hap_cols[i + 1]
        sample0 = h0.rsplit(".", 1)[0]
        sample1 = h1.rsplit(".", 1)[0]
        if sample0 != sample1:
            raise ValueError(f"Haplotype columns are not paired: {h0}, {h1}")
        samples.append(sample0)
    return samples


def _segments_to_loci(
    segs: pd.DataFrame,
    hap_cols: List[str],
    pop_map: Dict[str, int],
    index_offset: int = 0,
) -> Tuple[pd.DataFrame, da.Array]:
    """
    Convert segment rows to loci-level data and a dask ancestry array.

    The loci DataFrame has one row per segment (the segment start = reference
    locus).  The ancestry array has shape (n_segments, n_samples, n_ancestries)
    with hard integer counts (0/1/2) derived by summing the two haplotypes per
    sample.  Axis 2 follows the population *codes* of the file header, i.e.
    the order RFMix itself uses.

    Parameters
    ----------
    segs : pd.DataFrame
        Output of _read_msp_file, one row per segment.
    hap_cols : list of str
        Per-haplotype column names (Sample_1.0, Sample_1.1, …).
    pop_map : dict
        Population label → integer code (e.g. {"AFR": 0, "EUR": 1}).
    index_offset : int
        Global locus index offset for multi-chromosome datasets.

    Returns
    -------
    loci_df : DataFrame
        Columns: chromosome, physical_position (= spos), i.
    local_array : dask.array.Array
        Shape (n_segments, n_samples, n_ancestries), dtype int8.
    """
    pops = pops_by_code(pop_map)
    n_pops = len(pops)
    n_samples = len(_sample_names_from_hap_cols(hap_cols))
    n_segs = len(segs)

    hap_data = segs[hap_cols].to_numpy(dtype=np.int16)  # (n_segs, 2*n_samples)
    if hap_data.size and (hap_data.min() < 0 or hap_data.max() >= n_pops):
        raise ValueError(
            f"Ancestry codes must be in 0..{n_pops - 1} (populations {pops}); "
            f"found values in [{hap_data.min()}, {hap_data.max()}]."
        )
    hap0 = hap_data[:, 0::2]  # (n_segs, n_samples) — haplotype 0
    hap1 = hap_data[:, 1::2]  # (n_segs, n_samples) — haplotype 1
    local_np = counts_from_hap_codes(hap0, hap1, n_pops)  # int8

    local_array = da.from_array(
        local_np, chunks=(min(1024, max(n_segs, 1)), n_samples, n_pops)
    )

    loci_df = pd.DataFrame({
        "chromosome": pd.Categorical(segs["chrom"].astype(str)),
        "physical_position": segs["spos"].astype(np.int32).to_numpy(),
        "i": np.arange(index_offset, index_offset + n_segs, dtype=np.int64),
    })

    return loci_df, local_array


def _get_msp_prefixes(file_prefix: str, verbose: bool = True) -> List[Dict[str, str]]:
    """
    Find .msp.tsv (or .msp.tsv.gz) files under file_prefix and return a
    list of per-chromosome file maps (see :func:`get_prefixes`).
    """
    try:
        return get_prefixes(file_prefix, "msp", verbose)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No .msp.tsv files found under prefix: {file_prefix}"
        ) from None


def _read_Q_for_msp(fn: str) -> pd.DataFrame:
    """Read an RFMix ``.rfmix.Q`` file next to MSP output."""
    return read_rfmix_q(fn, add_chrom=True)


def extract_locus_ancestry(
    file_prefix: str,
    loci: pd.DataFrame,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    samples: Optional[Sequence[str]] = None,
    aggregate: bool = True,
) -> pd.DataFrame:
    """
    Query MSP hard-call local ancestry at selected SNP positions.

    This function scans MSP interval rows and returns ancestry at the requested
    loci without expanding the full genome into a dense variant-by-sample matrix.
    Intervals are treated as closed on both sides: ``spos <= pos <= epos``.
    """
    if chrom_col not in loci.columns or pos_col not in loci.columns:
        raise ValueError(
            f"loci must contain '{chrom_col}' and '{pos_col}' columns."
        )
    if loci.empty:
        return loci.copy()

    loci_work = loci.reset_index(drop=False).rename(columns={"index": "_locus_index"})
    loci_work["_chrom_norm"] = loci_work[chrom_col].map(lambda v: _normalize_chrom_label(str(v).strip()))
    loci_work["_pos_int"] = pd.to_numeric(loci_work[pos_col], errors="raise").astype(np.int64)
    target_chroms = set(loci_work["_chrom_norm"].astype(str))

    rows: List[dict] = []
    seen_loci = set()
    first_pop_labels: List[str] = []
    first_sample_count: Optional[int] = None

    for filemap in _get_msp_prefixes(file_prefix, verbose=False):
        segs_pd, hap_cols, pop_map = _read_msp_file(filemap["msp.tsv"])
        seg_chroms = segs_pd["chrom"].astype(str).map(lambda v: _normalize_chrom_label(v.strip()))
        common_chroms = sorted(target_chroms.intersection(set(seg_chroms.astype(str))))
        if not common_chroms:
            continue

        sample_names = _sample_names_from_hap_cols(hap_cols)
        pop_labels = pops_by_code(pop_map)
        if not first_pop_labels:
            first_pop_labels = pop_labels
        if samples is None:
            sample_idx = list(range(len(sample_names)))
            selected_samples = sample_names
        else:
            missing = [sample for sample in samples if sample not in sample_names]
            if missing:
                raise ValueError(f"Samples not found in MSP file: {missing}")
            sample_idx = [sample_names.index(sample) for sample in samples]
            selected_samples = list(samples)
        if first_sample_count is None:
            first_sample_count = len(selected_samples)

        n_pops = len(pop_labels)
        selected_hap_cols = []
        for sample_i in sample_idx:
            selected_hap_cols.extend(hap_cols[(2 * sample_i):(2 * sample_i + 2)])

        for chrom in common_chroms:
            seg_chr = segs_pd.loc[seg_chroms == chrom].sort_values("spos")
            target_chr = loci_work.loc[loci_work["_chrom_norm"] == chrom].copy()
            starts = seg_chr["spos"].to_numpy(dtype=np.int64)
            ends = seg_chr["epos"].to_numpy(dtype=np.int64)
            positions = target_chr["_pos_int"].to_numpy(dtype=np.int64)
            seg_idx = np.searchsorted(starts, positions, side="right") - 1
            safe_idx = np.clip(seg_idx, 0, max(len(seg_chr) - 1, 0))
            in_bounds = (
                (seg_idx >= 0)
                & (seg_idx < len(seg_chr))
                & (positions <= ends[safe_idx])
            )
            seg_chr_reset = seg_chr.reset_index(drop=True)

            for target_row, idx, matched in zip(target_chr.to_dict("records"), seg_idx, in_bounds):
                base = {
                    k: v for k, v in target_row.items()
                    if k not in {"_chrom_norm", "_pos_int"}
                }
                base["matched"] = bool(matched)
                seen_loci.add(base["_locus_index"])
                if not matched:
                    if aggregate:
                        base["n_samples"] = len(selected_samples)
                        base["n_haplotypes"] = 0
                        for pop in pop_labels:
                            base[f"{pop}_haplotypes"] = np.nan
                            base[f"{pop}_fraction"] = np.nan
                        rows.append(base)
                    else:
                        for sample in selected_samples:
                            sample_row = dict(base)
                            sample_row["sample_id"] = sample
                            for pop in pop_labels:
                                sample_row[f"{pop}_copies"] = np.nan
                            rows.append(sample_row)
                    continue

                calls = seg_chr_reset.loc[int(idx), selected_hap_cols].to_numpy(dtype=np.int16)
                calls = calls.reshape(len(selected_samples), 2)
                if aggregate:
                    flat = calls.reshape(-1)
                    counts = np.bincount(flat, minlength=n_pops)
                    total = int(counts.sum())
                    base["n_samples"] = len(selected_samples)
                    base["n_haplotypes"] = total
                    for pop, count in zip(pop_labels, counts):
                        base[f"{pop}_haplotypes"] = int(count)
                        base[f"{pop}_fraction"] = float(count / total) if total else np.nan
                    rows.append(base)
                else:
                    for sample, sample_calls in zip(selected_samples, calls):
                        sample_row = dict(base)
                        sample_row["sample_id"] = sample
                        sample_counts = np.bincount(sample_calls, minlength=n_pops)
                        for pop, count in zip(pop_labels, sample_counts):
                            sample_row[f"{pop}_copies"] = int(count)
                        rows.append(sample_row)

    missing_loci = loci_work.loc[~loci_work["_locus_index"].isin(seen_loci)]
    for target_row in missing_loci.to_dict("records"):
        base = {
            k: v for k, v in target_row.items()
            if k not in {"_chrom_norm", "_pos_int"}
        }
        base["matched"] = False
        if aggregate:
            base["n_samples"] = first_sample_count if first_sample_count is not None else np.nan
            base["n_haplotypes"] = 0
            for pop in first_pop_labels:
                base[f"{pop}_haplotypes"] = np.nan
                base[f"{pop}_fraction"] = np.nan
            rows.append(base)
        else:
            selected_samples = list(samples) if samples is not None else []
            if not selected_samples:
                rows.append(base)
            for sample in selected_samples:
                sample_row = dict(base)
                sample_row["sample_id"] = sample
                for pop in first_pop_labels:
                    sample_row[f"{pop}_copies"] = np.nan
                rows.append(sample_row)

    result = pd.DataFrame(rows)
    if "_locus_index" in result.columns:
        sort_cols = ["_locus_index"]
        if "sample_id" in result.columns:
            sort_cols.append("sample_id")
        result = result.sort_values(sort_cols).reset_index(drop=True)
        result = result.drop(columns=["_locus_index"])
    return result


def read_rfmix(
    file_prefix: str,
    g_anc: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    chrom: Optional[str] = None,
    read_q: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], da.Array]:
    """
    Read RFMix `.msp.tsv` files into a loci DataFrame and a Dask ancestry array.

    This is the **recommended default reader** for RFMix output. The `.msp.tsv`
    format stores piecewise-constant hard ancestry calls as genomic segments and
    is roughly **2,000× smaller** than the corresponding `.fb.tsv` forward-backward
    matrix. Reading it directly eliminates the costly binary-conversion step and is
    sufficient for the vast majority of local ancestry analyses (GWAS, admixture
    mapping, QC, visualization).

    Use :func:`read_rfmix_fb` instead when you explicitly require posterior
    probability values from the forward-backward matrix.

    Parameters
    ----------
    file_prefix : str
        Directory, file, or path prefix under which ``.msp.tsv`` files live.
    g_anc : DataFrame, optional
        Pre-loaded global ancestry DataFrame.  When provided its ancestry
        columns are reordered to match ``local_array`` and it is returned.
        When :data:`None` and ``read_q`` is false, the second return value is
        :data:`None`.
    verbose : bool, default True
        Print progress information.
    chrom : str, optional
        Restrict reading to a single chromosome (with or without ``chr`` prefix).
    read_q : bool, default True
        When ``g_anc`` is not supplied, read neighboring ``.rfmix.Q`` files when
        present and return their concatenated global ancestry table.

    Returns
    -------
    loci_df : DataFrame
        Columns: ``chromosome``, ``physical_position``, ``i``.
        One row per RFMix ancestry segment boundary.
    g_anc : DataFrame or None
        Global ancestry with ancestry columns in the same order as axis 2 of
        ``local_array``, or :data:`None`.
    local_array : dask.array.Array
        Shape ``(n_segments, n_samples, n_ancestries)``, dtype ``int8``.
        Hard ancestry counts (0/1/2).  Axis 2 follows the population codes of
        the ``.msp.tsv`` header (``#Subpopulation order/codes``), which is the
        order RFMix uses everywhere.

    Notes
    -----
    Local ancestry is at **segment resolution** — each row is the start
    position of one RFMix ancestry segment.  Use :func:`interpolate_array`
    with ``method='stepwise'`` to expand segments onto a denser variant grid,
    or :func:`extract_locus_ancestry` to query individual positions.

    Examples
    --------
    >>> loci, g_anc, admix = read_rfmix("data/rfmix_out/")
    >>> print(loci.shape, admix.shape)
    (1629, 3) (1629, 81, 2)
    """
    maybe_report_gpu(verbose)

    fn = filter_file_maps_by_chrom(
        _get_msp_prefixes(file_prefix, verbose), chrom, kind="MSP"
    )

    pbar = tqdm(desc="Reading MSP files", total=len(fn), disable=not verbose)
    results = _read_file(fn, lambda f: _read_msp_file(f["msp.tsv"]), pbar)
    pbar.close()

    pops = check_pop_order_consistent(
        [pops_by_code(pop_map) for _, _, pop_map in results], kind="MSP"
    )

    if g_anc is None and read_q:
        q_maps = [f for f in fn if "rfmix.Q" in f]
        if q_maps:
            pbar = tqdm(desc="Reading Q files", total=len(q_maps), disable=not verbose)
            q_dfs = _read_file(q_maps, lambda f: _read_Q_for_msp(f["rfmix.Q"]), pbar)
            pbar.close()
            g_anc = pd.concat(q_dfs, axis=0, ignore_index=True)

    if g_anc is not None:
        g_anc = align_g_anc_columns(g_anc, pops)

    index_offset = 0
    loci_dfs = []
    local_arrays = []
    for segs, hap_cols, pop_map in results:
        loci_df_i, local_i = _segments_to_loci(segs, hap_cols, pop_map, index_offset)
        loci_dfs.append(loci_df_i)
        local_arrays.append(local_i)
        index_offset += len(segs)

    loci_df = pd.concat(loci_dfs, axis=0, ignore_index=True)
    local_array = da.concatenate(local_arrays, axis=0)

    loci_df, g_anc = maybe_to_backend_frames(loci_df, g_anc)
    return loci_df, g_anc, local_array
