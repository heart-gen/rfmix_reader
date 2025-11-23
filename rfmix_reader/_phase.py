"""
local_ancestry_phasing.py

Phasing corrections for local ancestry haplotypes, inspired by gnomix.

The overall strategy is adapted from the phasing utilities in gnomix's
`phasing.py` (notably:
    - find_hetero_regions
    - get_ref_map / find_ref
    - correct_phase_error
which use reference haplotypes and tail-swapping to correct phasing).

This module provides a lightweight, NumPy-based implementation that:
    1. Identifies heterozygous ancestry regions (M != P).
    2. Compares haplotypes to two references in sliding windows.
    3. Builds a "phase track" of where to flip suffixes.
    4. Applies tail flips to obtain phase-corrected haplotypes.

Author: (your name / HEART-GeN)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from dask.array import Array as DaskArray
import dask.array as da

from cyvcf2 import VCF
import pandas as pd

ArrayLike = np.ndarray

@dataclass
class PhasingConfig:
    """
    Configuration for local ancestry phase correction.

    Parameters
    ----------
    window_size : int
        Number of SNPs per phasing window. Tail flips occur at window
        boundaries. Larger windows -> more smoothing, fewer spurious flips.
    min_block_len : int
        Minimum length (in SNPs) for a heterozygous block to consider for
        phasing corrections. Very short blocks are often uninformative.
    max_mismatch_frac : float
        If both references mismatch a window badly, we treat it as
        uninformative (no strong evidence to flip).
    verbose : bool
        If True, prints basic diagnostics per sample / region.
    """

    window_size: int = 50
    min_block_len: int = 20
    max_mismatch_frac: float = 0.5
    verbose: bool = False


# ---------------------------------------------------------------------------
# Heterozygous region detection (adapted from gnomix `find_hetero_regions`)
# ---------------------------------------------------------------------------

def find_heterozygous_blocks(
    hap0: ArrayLike, hap1: ArrayLike, min_block_len: int = 1,
) -> List[slice]:
    """
    Find contiguous blocks where hap0 != hap1 (heterozygous ancestry).

    This follows the same conceptual idea as gnomix's `find_hetero_regions`,
    which locates regions where the two haplotypes carry different ancestry
    labels and where phasing is actually informative.

    Parameters
    ----------
    hap0, hap1 : (L,) int
        Ancestry-coded haplotypes for a single individual.
    min_block_len : int
        Minimum number of SNPs for a block to be returned.

    Returns
    -------
    blocks : list of slice
        List of index slices (start:end) for heterozygous regions.
    """
    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)
    if hap0.shape != hap1.shape:
        raise ValueError("hap0 and hap1 must have the same shape.")

    het = hap0 != hap1
    if not np.any(het):
        return []

    # boundaries where hetero mask changes
    boundaries = np.concatenate(
        ([0], np.where(het[:-1] != het[1:])[0] + 1, [len(het)])
    )

    blocks: List[slice] = []
    for b in range(len(boundaries) - 1):
        start, end = boundaries[b], boundaries[b + 1]
        if het[start] and (end - start) >= min_block_len:
            blocks.append(slice(start, end))

    return blocks


# ---------------------------------------------------------------------------
# Windowing and reference assignment
# (conceptually adapted from gnomix `get_ref_map` / `find_ref`)
# ---------------------------------------------------------------------------

def window_slices(n: int, window_size: int) -> List[slice]:
    """
    Yield index slices of length `window_size` across 0..n.

    The last window may be shorter if n is not a multiple of window_size.
    """
    return [slice(i, min(i + window_size, n)) for i in range(0, n, window_size)]


def assign_reference_per_window(
    hap: ArrayLike,
    ref1: ArrayLike,
    ref2: ArrayLike,
    window_size: int,
    max_mismatch_frac: float,
) -> np.ndarray:
    """
    For each window, decide whether `hap` matches ref1 or ref2 better.

    This is analogous in spirit to gnomix's `get_ref_map`, which tracks
    which of two reference haplotypes a given haplotype is following.

    Parameters
    ----------
    hap, ref1, ref2 : (L,) int
        Haplotype of interest and two reference haplotypes (same length).
    window_size : int
        Number of SNPs per phasing window.
    max_mismatch_frac : float
        If both refs mismatch > this fraction of sites, window is treated
        as uninformative and assigned 0 (no strong ref).

    Returns
    -------
    ref_track : (W,) int
        For each window w:
          0 : ambiguous / low-confidence
          1 : matches ref1 better
          2 : matches ref2 better
    """
    hap = np.asarray(hap)
    ref1 = np.asarray(ref1)
    ref2 = np.asarray(ref2)

    if not (hap.shape == ref1.shape == ref2.shape):
        raise ValueError("hap, ref1, ref2 must have the same shape.")

    n = hap.shape[0]
    wslices = window_slices(n, window_size)
    ref_track = np.zeros(len(wslices), dtype=np.int8)

    for w_idx, sl in enumerate(wslices):
        h_win = hap[sl]
        r1_win = ref1[sl]
        r2_win = ref2[sl]

        # Ignore positions where references are missing (use -1 as missing if needed)
        mask_valid = (r1_win >= 0) & (r2_win >= 0)
        if not np.any(mask_valid):
            ref_track[w_idx] = 0
            continue

        h_win = h_win[mask_valid]
        r1_win = r1_win[mask_valid]
        r2_win = r2_win[mask_valid]

        mism1 = np.mean(h_win != r1_win)
        mism2 = np.mean(h_win != r2_win)

        # If both mismatch too much, treat as ambiguous
        if mism1 >= max_mismatch_frac and mism2 >= max_mismatch_frac:
            ref_track[w_idx] = 0
        elif mism1 <= mism2:
            ref_track[w_idx] = 1
        else:
            ref_track[w_idx] = 2

    return ref_track


# ---------------------------------------------------------------------------
# Phase track and tail flipping
# (adapted from gnomix `correct_phase_error` / `track_switch`)
# ---------------------------------------------------------------------------

def build_phase_track_from_ref(ref_track: np.ndarray) -> np.ndarray:
    """
    Build a window-level "phase flip track" from reference assignments.

    The idea is: if the reference assignment for a haplotype changes
    often (e.g. 1 -> 2 or 2 -> 1), that indicates a possible phase flip
    between the two haplotypes.

    We construct a cumulative 0/1 track where each change in ref assignment
    toggles the phase. This is similar in spirit to how gnomix uses
    reference maps and `track_switch` to decide where to swap suffixes.

    Parameters
    ----------
    ref_track : (W,) int
        Window-level reference assignments: 0, 1, or 2.

    Returns
    -------
    phase_track : (W,) int
        0/1 flag per window. A change from 0->1 or 1->0 indicates that
        windows from that point onward should have M/P swapped.
    """
    ref_track = np.asarray(ref_track)
    W = ref_track.shape[0]

    phase_track = np.zeros(W, dtype=np.int8)

    # Only consider windows with a clear reference (1 or 2).
    # We track transitions among these labels; ambiguous windows (0)
    # just inherit the current phase state.
    last_ref = 0
    current_phase = 0

    for w in range(W):
        ref = ref_track[w]
        if ref in (1, 2):
            if last_ref == 0:
                # first informative window
                last_ref = ref
            elif ref != last_ref:
                # reference changed -> toggle phase downstream
                current_phase ^= 1
                last_ref = ref
        phase_track[w] = current_phase

    return phase_track


def apply_phase_track(
    hap0: ArrayLike, hap1: ArrayLike, phase_track: np.ndarray, window_size: int,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Apply tail flips between hap0 and hap1 according to phase_track.

    This is analogous to gnomix's `correct_phase_error`, which uses a
    0/1 "track" over windows to decide where to swap suffixes between the
    two haplotypes.

    Parameters
    ----------
    hap0, hap1 : (L,) int
        Ancestry-coded haplotypes to be corrected.
    phase_track : (W,) int
        0/1 flags per window; when it changes 0->1 or 1->0, tails are swapped.
    window_size : int
        Number of SNPs per window.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) int
        Phase-corrected haplotypes.
    """
    hap0 = np.asarray(hap0).copy()
    hap1 = np.asarray(hap1).copy()
    W = phase_track.shape[0]

    # Identify window boundaries where phase state changes
    change_points = np.where(np.diff(phase_track) != 0)[0] + 1  # window indices
    # Convert window indices to SNP indices
    flip_positions = [w * window_size for w in change_points if w < W]

    for pos in flip_positions:
        # swap tails from pos onward
        tmp = hap0[pos:].copy()
        hap0[pos:] = hap1[pos:]
        hap1[pos:] = tmp

    return hap0, hap1


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def phase_local_ancestry_sample(
    hap0: ArrayLike, hap1: ArrayLike, ref0: ArrayLike, ref1: ArrayLike,
    config: Optional[PhasingConfig] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Perform gnomix-style phasing corrections for a single individual.

    Parameters
    ----------
    hap0, hap1 : (L,) int
        Two haplotypes for the individual (e.g. maternal/paternal) with
        ancestry labels at each locus.
    ref0, ref1 : (L,) int
        Reference haplotypes. In simulations, these can be the true
        haplotypes. In real data, they can be chosen from a reference panel
        or from other individuals with similar ancestry.
    config : PhasingConfig, optional
        Configuration for window size, thresholds, etc.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) int
        Phase-corrected haplotypes.
    """
    if config is None:
        config = PhasingConfig()

    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)
    ref0 = np.asarray(ref0)
    ref1 = np.asarray(ref1)

    if not (hap0.shape == hap1.shape == ref0.shape == ref1.shape):
        raise ValueError("All haplotypes must have the same length.")

    L = hap0.shape[0]
    if L == 0:
        return hap0.copy(), hap1.copy()

    het_blocks = find_heterozygous_blocks(
        hap0, hap1, min_block_len=config.min_block_len
    )

    if config.verbose:
        print(f"[phase_local_ancestry_sample] {len(het_blocks)} heterozygous blocks")

    hap0_corr = hap0.copy()
    hap1_corr = hap1.copy()

    for block_idx, block in enumerate(het_blocks):
        start, end = block.start, block.stop

        if config.verbose:
            print(f"  - block {block_idx}: {start}..{end} (len={end-start})")

        # Extract block
        h0_blk = hap0_corr[start:end]
        h1_blk = hap1_corr[start:end]
        r0_blk = ref0[start:end]
        r1_blk = ref1[start:end]

        # For simplicity, we build reference assignment based on one haplotype,
        # e.g., `h0_blk`. You can also combine information from both.
        ref_track = assign_reference_per_window(
            hap=h0_blk,
            ref1=r0_blk,
            ref2=r1_blk,
            window_size=config.window_size,
            max_mismatch_frac=config.max_mismatch_frac,
        )

        phase_track = build_phase_track_from_ref(ref_track)

        # Apply flips inside this block only
        #  - we restrict to the [start:end] region
        local_L = end - start
        local_W = int(np.ceil(local_L / config.window_size))
        # Ensure phase_track has the correct number of windows
        if phase_track.shape[0] != local_W:
            # Truncate or pad with final value if needed (safety)
            if phase_track.shape[0] > local_W:
                phase_track = phase_track[:local_W]
            else:
                pad_val = phase_track[-1] if phase_track.size > 0 else 0
                phase_track = np.pad(
                    phase_track,
                    (0, local_W - phase_track.shape[0]),
                    constant_values=pad_val,
                )

        h0_blk_corr, h1_blk_corr = apply_phase_track(
            h0_blk, h1_blk, phase_track, config.window_size
        )

        hap0_corr[start:end] = h0_blk_corr
        hap1_corr[start:end] = h1_blk_corr

    return hap0_corr, hap1_corr


def load_sample_annotations(
    annot_path: str, sep: str = r"\s+", col_sample: str = "sample_id",
    col_group: str = "group",
) -> pd.DataFrame:
    """
    Load sample annotation file mapping sample_id -> group (e.g., ancestry).

    Expected format: two columns (no header) by default:
        sample_id   group

    Example:
        HG00271 EUR
        HG00276 EUR

    Parameters
    ----------
    annot_path : str
        Path to text file with sample annotations.
    sep : str, default=r"\\s+"
        Field separator pattern (passed to pandas.read_csv).
    col_sample : str, default "sample_id"
        Column name to assign to sample IDs.
    col_group : str, default "group"
        Column name to assign to group labels (e.g., EUR/AFR/etc.).

    Returns
    -------
    annot : pandas.DataFrame
        DataFrame with columns [col_sample, col_group].
    """
    annot = pd.read_csv(
        annot_path,
        sep=sep,
        header=None,
        names=[col_sample, col_group],
        dtype={col_sample: str, col_group: str},
    )
    return annot


def choose_reference_samples(
    annot: pd.DataFrame, group0: Optional[str] = None,
    group1: Optional[str] = None, col_sample: str = "sample_id",
    col_group: str = "group",
) -> Tuple[str, str]:
    """
    Choose two reference sample IDs from annotation table.

    Parameters
    ----------
    annot : DataFrame
        Output of `load_sample_annotations`.
    group0, group1 : str, optional
        Group labels to choose ref0 and ref1 from.
        If group1 is None, both references are drawn from group0.
        If group0 is None, uses the most frequent group in `annot`.
    col_sample, col_group : str
        Column names for sample ID and group label.

    Returns
    -------
    ref0_id, ref1_id : str
        Sample IDs to be used as reference haplotypes.
    """
    df = annot

    if group0 is None:
        # pick most frequent group
        group0 = df[col_group].value_counts().idxmax()
    if group1 is None:
        group1 = group0

    df0 = df[df[col_group] == group0]
    df1 = df[df[col_group] == group1]

    if df0.empty:
        raise ValueError(f"No samples found for group0='{group0}'.")
    if df1.empty:
        raise ValueError(f"No samples found for group1='{group1}'.")

    # simple choice: take first sample in each group
    ref0_id = df0[col_sample].iloc[0]
    # ensure ref1 is not identical if groups are same and we have >=2 samples
    if group1 == group0 and len(df1) > 1:
        ref1_id = df1[col_sample].iloc[1]
    else:
        ref1_id = df1[col_sample].iloc[0]

    if ref0_id == ref1_id:
        raise ValueError("ref0 and ref1 ended up being the same sample; "
                         "provide distinct groups or more samples.")

    return ref0_id, ref1_id


def build_reference_haplotypes_from_vcf(
    vcf_path: str, annot_path: str, chrom: str, positions: np.ndarray,
    group0: Optional[str] = None, group1: Optional[str] = None,
    hap_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build reference haplotypes (ref0, ref1) from an indexed VCF using cyvcf2
    and a sample annotation file.

    The reference haplotypes are haploid sequences (one allele per site),
    which are then used as `ref0` and `ref1` in gnomix-style phase
    correction. The logic is:

      1. Load annotations and choose ref0/ref1 sample IDs.
      2. Map those IDs to indices in the VCF's sample list.
      3. For each variant on `chrom`, if its position is in `positions`,
         extract the chosen haploid allele for ref0 and ref1.

    NOTE:
      - This is *not* ancestry-coded; it's allele-coded. For phasing-correction
        of local ancestry, you're using reference haplotype *patterns* as a
        proxy, not true ancestry labels.

    Parameters
    ----------
    vcf_path : str
        Path to bgzipped, indexed VCF/BCF file (for cyvcf2).
    annot_path : str
        Path to sample annotation file (see `load_sample_annotations`).
    chrom : str
        Chromosome name to extract (e.g., "1", "chr1").
    positions : array-like of int
        Positions (1-based bp) for which we want reference alleles, length L.
    group0, group1 : str, optional
        Group labels to choose ref0 and ref1 from. If None, defaults to the
        most frequent group in the annotation file.
    hap_index : int, default 0
        Which haploid allele to take (0 or 1) from the genotype. For phased
        calls, this is the first allele in GT like 0|1.

    Returns
    -------
    ref0_hap, ref1_hap : (L,) int
        Haploid reference haplotypes as allele codes (0, 1, 2, or -1 for
        missing). Positions not found in the VCF are set to -1.
    """
    positions = np.asarray(positions, dtype=np.int64)
    L = positions.shape[0]

    # 1. Load annotations and choose reference sample IDs
    annot = load_sample_annotations(annot_path)
    ref0_id, ref1_id = choose_reference_samples(
        annot, group0=group0, group1=group1
    )

    vcf = VCF(vcf_path)
    sample_to_idx = {sid: i for i, sid in enumerate(vcf.samples)}

    if ref0_id not in sample_to_idx:
        raise ValueError(f"Reference sample ref0_id '{ref0_id}' not found in VCF.")
    if ref1_id not in sample_to_idx:
        raise ValueError(f"Reference sample ref1_id '{ref1_id}' not found in VCF.")

    ref0_idx = sample_to_idx[ref0_id]
    ref1_idx = sample_to_idx[ref1_id]

    # 2. Prepare arrays: -1 means "missing / not found"
    ref0_hap = np.full(L, -1, dtype=np.int8)
    ref1_hap = np.full(L, -1, dtype=np.int8)

    # map position -> index in positions array
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}

    # 3. Iterate over variants on this chromosome
    region_str = str(chrom)  # cyvcf2 uses "1" or "chr1" consistent with VCF
    for variant in vcf(region_str):
        pos = int(variant.POS)
        if pos not in pos_to_idx:
            continue
        idx = pos_to_idx[pos]

        gts = variant.genotypes  # shape (n_samples, 3 or 4) -> [a1, a2, phased, ...]
        gt0 = gts[ref0_idx]
        gt1 = gts[ref1_idx]

        # treat negative or missing as -1
        # gt = [allele1, allele2, phased_flag, ...]
        a0 = gt0[hap_index] if gt0[hap_index] >= 0 else -1
        a1 = gt1[hap_index] if gt1[hap_index] >= 0 else -1

        ref0_hap[idx] = a0
        ref1_hap[idx] = a1

    return ref0_hap, ref1_hap

# ---------------------------------------------------------------------------
# Optional: simple driver for a sample + metrics with truth
# ---------------------------------------------------------------------------

def count_switch_errors(
    M_pred: ArrayLike, P_pred: ArrayLike, M_true: ArrayLike, P_true: ArrayLike,
) -> int:
    """
    Count minimal number of phase switches needed to turn (M_pred, P_pred)
    into (M_true, P_true).

    This is functionally similar to gnomix's `find_switches`, but implemented
    here in a compact way.

    Parameters
    ----------
    M_pred, P_pred : (L,) int
        Predicted ancestry haplotypes.
    M_true, P_true : (L,) int
        Ground-truth ancestry haplotypes.

    Returns
    -------
    n_switches : int
        Number of phase switch points.
    """
    M_pred = np.asarray(M_pred).copy()
    P_pred = np.asarray(P_pred).copy()
    M_true = np.asarray(M_true)
    P_true = np.asarray(P_true)

    if not (M_pred.shape == P_pred.shape == M_true.shape == P_true.shape):
        raise ValueError("All haplotypes must have the same shape.")

    n_switches = 0
    L = M_pred.shape[0]
    for i in range(L):
        if M_pred[i] != M_true[i]:
            # swap tails from i onward
            M_tmp = M_pred[i:].copy()
            M_pred[i:] = P_pred[i:]
            P_pred[i:] = M_tmp
            n_switches += 1

    if not (np.array_equal(M_pred, M_true) and np.array_equal(P_pred, P_true)):
        raise RuntimeError("Phase error correction did not align with truth.")

    return n_switches


def phase_local_ancestry_sample_from_vcf(
    hap0: np.ndarray,
    hap1: np.ndarray,
    positions: np.ndarray,
    chrom: str,
    ref_vcf_path: str,
    sample_annot_path: str,
    group0: Optional[str] = None,
    group1: Optional[str] = None,
    config: Optional[PhasingConfig] = None,
    hap_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-correct local ancestry haplotypes using gnomix-style logic,
    with reference haplotypes drawn from an indexed VCF.

    This is a convenience wrapper that:
      1. Loads sample annotations to choose reference samples (ref0, ref1).
      2. Builds haploid reference haplotypes for those samples at the
         positions of interest using `cyvcf2`.
      3. Calls `phase_local_ancestry_sample` with these references to
         perform gnomix-style tail-flip corrections.

    Phasing logic itself (heterozygous region detection, windowed
    reference mapping, and tail flipping) is adapted from gnomix
    `phasing.py` functions:
        - `find_hetero_regions`
        - `get_ref_map` / `find_ref`
        - `correct_phase_error` / `track_switch`.

    Parameters
    ----------
    hap0, hap1 : (L,) int
        Local ancestry haplotypes for the individual (e.g. 0/1/2 for
        ancestry labels) on a given chromosome.
    positions : (L,) int
        1-based bp positions corresponding to hap0/hap1 entries.
    chrom : str
        Chromosome name to use when querying the VCF (e.g. "1" or "chr1").
    ref_vcf_path : str
        Path to bgzipped, indexed reference VCF/BCF file.
    sample_annot_path : str
        Path to sample annotation file (sample_id + group label).
    group0, group1 : str, optional
        Group labels used to choose ref0 and ref1 samples from the
        annotation file. If None, the most frequent group is used for
        both references (two distinct samples from that group).
    config : PhasingConfig, optional
        Configuration for phasing (window size, thresholds, verbosity).
    hap_index : int, default 0
        Which haploid allele to take from the reference VCF genotypes.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) int
        Phase-corrected local ancestry haplotypes.
    """
    if config is None:
        config = PhasingConfig()

    positions = np.asarray(positions, dtype=np.int64)
    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)

    if hap0.shape != hap1.shape or hap0.shape[0] != positions.shape[0]:
        raise ValueError("hap0, hap1, and positions must all have length L.")

    # Build reference haplotypes from VCF
    ref0_hap, ref1_hap = build_reference_haplotypes_from_vcf(
        vcf_path=ref_vcf_path,
        annot_path=sample_annot_path,
        chrom=chrom,
        positions=positions,
        group0=group0,
        group1=group1,
        hap_index=hap_index,
    )

    # Run gnomix-style phasing correction
    hap0_corr, hap1_corr = phase_local_ancestry_sample(
        hap0=hap0,
        hap1=hap1,
        ref0=ref0_hap,
        ref1=ref1_hap,
        config=config,
    )

    return hap0_corr, hap1_corr


def build_hap_labels_from_rfmix(
    X_raw: DaskArray | np.ndarray,
    hap_index: np.ndarray,
    sample_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use the original RFMix matrix + hap_index to reconstruct hap0/hap1
    ancestry labels for a single sample.

    Parameters
    ----------
    X_raw : (L, n_cols) dask.array or np.ndarray
        Original RFMix matrix before summing haps.
        Each column is something like "ancestry a, hap h, sample s".
    hap_index : (n_samples, n_anc, 2) np.ndarray
        From _subset_populations; hap_index[s, a, 0/1] gives the column
        index in X_raw for hap0 / hap1 of sample s and ancestry a.
    sample_idx : int
        Index of the sample to reconstruct.

    Returns
    -------
    hap0_labels, hap1_labels : (L,) int
        Per-locus ancestry labels (0..n_anc-1) for hap0 and hap1.
        -1 indicates missing (no ancestry column > 0 at that locus).
    """
    # Make sure we’re working with numpy arrays for the slice we need
    if hasattr(X_raw, "compute"):
        X = X_raw[:, :].compute()  # or slice more carefully if memory is tight
    else:
        X = np.asarray(X_raw)

    n_loci, n_cols = X.shape
    _, n_anc, _ = hap_index.shape

    # Column indices for this sample’s hap0 and hap1 across ancestries
    hap_idx_s = hap_index[sample_idx]            # (n_anc, 2)
    cols_h0   = hap_idx_s[:, 0]                  # (n_anc,)
    cols_h1   = hap_idx_s[:, 1]                  # (n_anc,)

    # Extract per-hap matrices: (L, n_anc)
    # X[l, cols_h0[a]] ~ probability/indicator “hap0 is ancestry a”
    H0 = X[:, cols_h0]    # (L, n_anc)
    H1 = X[:, cols_h1]    # (L, n_anc)

    # Ancestry label per hap = argmax over ancestries
    hap0_labels = np.argmax(H0, axis=1).astype(np.int16)
    hap1_labels = np.argmax(H1, axis=1).astype(np.int16)

    # Optionally mark loci that are all-zero (no assignment) as -1
    mask0 = (H0.sum(axis=1) == 0)
    mask1 = (H1.sum(axis=1) == 0)
    hap0_labels[mask0] = -1
    hap1_labels[mask1] = -1

    return hap0_labels, hap1_labels


def combine_haps_to_counts(
    hap0: np.ndarray,
    hap1: np.ndarray,
    n_anc: int,
) -> np.ndarray:
    """
    Combine haplotype-level ancestry labels back into summed counts (0,1,2).
    """
    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)
    if hap0.shape != hap1.shape:
        raise ValueError("hap0 and hap1 must have same shape.")

    L = hap0.shape[0]
    out = np.zeros((L, n_anc), dtype=np.int8)

    idx = np.arange(L)

    valid0 = (hap0 >= 0) & (hap0 < n_anc)
    out[idx[valid0], hap0[valid0]] += 1

    valid1 = (hap1 >= 0) & (hap1 < n_anc)
    out[idx[valid1], hap1[valid1]] += 1

    return out


def phase_admix_sample_from_vcf_with_index(
    admix_sample: np.ndarray,           # (L, A) counts, mostly for shape / A
    X_raw: DaskArray | np.ndarray,      # (L, n_cols) raw RFMix matrix
    hap_index: np.ndarray,              # (n_samples, A, 2)
    sample_idx: int,
    positions: np.ndarray,              # (L,) positions
    chrom: str,
    ref_vcf_path: str,
    sample_annot_path: str,
    config: PhasingConfig,
    group0: str | None = None,
    group1: str | None = None,
    hap_index_in_vcf: int = 0,
) -> np.ndarray:
    """
    Phase-correct local ancestry for one sample, using hap_index to recover
    hap0/hap1 from the original RFMix matrix.

    Steps:
      1. Use X_raw + hap_index to build hap0/hap1 ancestry labels.
      2. Run gnomix-style phasing (phase_local_ancestry_sample_from_vcf).
      3. Recombine corrected hap0/hap1 into 0/1/2 counts.

    Parameters
    ----------
    admix_sample : (L, A) int
        Summed local ancestry (0/1/2) for this sample; used for L and A.
    X_raw : (L, n_cols)
        Original RFMix hap-by-ancestry matrix.
    hap_index : (n_samples, A, 2)
        Column mapping output by _subset_populations.
    sample_idx : int
        Sample index (0..n_samples-1).
    positions, chrom, ref_vcf_path, sample_annot_path, config, group0, group1
        As before.
    hap_index_in_vcf : int
        Which haploid allele to take from the reference VCF (0 or 1).

    Returns
    -------
    admix_corr : (L, A) int
        Phase-corrected summed ancestry counts for this sample.
    """
    L, A = admix_sample.shape

    # 1. reconstruct hap0/hap1 ancestry labels from raw RFMix output
    hap0, hap1 = build_hap_labels_from_rfmix(
        X_raw=X_raw,
        hap_index=hap_index,
        sample_idx=sample_idx,
    )

    if hap0.shape[0] != L:
        raise ValueError("Length of hap0/hap1 does not match admix_sample.")

    # 2. gnomix-style phasing using reference VCF
    hap0_corr, hap1_corr = phase_local_ancestry_sample_from_vcf(
        hap0=hap0,
        hap1=hap1,
        positions=positions,
        chrom=chrom,
        ref_vcf_path=ref_vcf_path,
        sample_annot_path=sample_annot_path,
        group0=group0,
        group1=group1,
        config=config,
        hap_index=hap_index_in_vcf,
    )

    # 3. recombine corrected haplotypes to summed counts
    admix_corr = combine_haps_to_counts(hap0_corr, hap1_corr, n_anc=A)
    return admix_corr


def phase_admix_dask_with_index(
    admix: DaskArray,                # (L, S, A) summed counts
    X_raw: DaskArray | np.ndarray,   # (L, n_cols) raw RFMix matrix
    hap_index: np.ndarray,           # (S, A, 2)
    positions: np.ndarray,           # (L,)
    chrom: str,
    ref_vcf_path: str,
    sample_annot_path: str,
    config: PhasingConfig,
    group0: str | None = None,
    group1: str | None = None,
) -> DaskArray:
    """
    Phase-correct all samples in a (L, S, A) admix array using hap_index
    and the original RFMix matrix.
    """
    n_loci, n_samples, n_anc = admix.shape
    corrected = []

    for s in range(n_samples):
        admix_s = admix[:, s, :].compute()  # (L, A)
        admix_s_corr = phase_admix_sample_from_vcf_with_index(
            admix_sample=admix_s,
            X_raw=X_raw,
            hap_index=hap_index,
            sample_idx=s,
            positions=positions,
            chrom=chrom,
            ref_vcf_path=ref_vcf_path,
            sample_annot_path=sample_annot_path,
            config=config,
            group0=group0,
            group1=group1,
        )
        corrected.append(admix_s_corr[None, :, :])  # (1, L, A)

    corrected_np = np.concatenate(corrected, axis=0)       # (S, L, A)
    corrected_np = np.transpose(corrected_np, (1, 0, 2))   # (L, S, A)

    admix_corr = da.from_array(corrected_np, chunks=admix.chunksize)
    return admix_corr
