"""
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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from cyvcf2 import VCF

import dask
import dask.array as da
from dask.array import Array as DaskArray

ArrayLike = np.ndarray

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PhasingConfig:
    """
    Configuration for local ancestry phase correction.

    Parameters
    ----------
    window_size : int, default 50
        Number of SNPs per phasing window. Tail flips occur only at window
        boundaries. Larger windows -> more smoothing, fewer spurious flips.
    min_block_len : int, default 20
        Minimum length (in SNPs) for a heterozygous block to consider for
        phasing corrections. Very short blocks are often uninformative.
    max_mismatch_frac : float, default 0.5
        If both references mismatch a window by more than this fraction of
        sites, the window is treated as uninformative (no strong evidence to
        flip).
    verbose : bool, default False
        If True, prints basic diagnostics per sample / region.
    """
    window_size: int = 50
    min_block_len: int = 20
    max_mismatch_frac: float = 0.5
    verbose: bool = False

# ============================================================================
# Heterozygous region detection
# ============================================================================

def find_heterozygous_blocks(
    hap0: ArrayLike, hap1: ArrayLike, min_block_len: int = 1,
) -> List[slice]:
    """
    Find contiguous blocks where ``hap0 != hap1`` (heterozygous ancestry).

    This follows the same conceptual idea as gnomix's ``find_hetero_regions``,
    which locates regions where the two haplotypes carry different ancestry
    labels and where phasing is actually informative.

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
        Ancestry-coded haplotypes for a single individual.
    min_block_len : int, default 1
        Minimum number of SNPs for a block to be returned. Very short
        heterozygous segments are usually noise and do not provide reliable
        evidence for phase correction.

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

    # boundaries where heterozygous mask changes
    boundaries = np.concatenate(
        ([0], np.where(het[:-1] != het[1:])[0] + 1, [len(het)])
    )

    blocks: List[slice] = []
    for b in range(len(boundaries) - 1):
        start, end = boundaries[b], boundaries[b + 1]
        if het[start] and (end - start) >= min_block_len:
            blocks.append(slice(start, end))

    return blocks

# ============================================================================
# Windowing and reference assignment
# ============================================================================

def window_slices(n: int, window_size: int) -> List[slice]:
    """
    Generate contiguous index slices of length ``window_size``.

    Parameters
    ----------
    n : int
        Total number of loci (0..n-1).
    window_size : int
        Length of each window in SNPs.

    Returns
    -------
    slices : list of slice
        Slices covering ``[0, n)``. The last window may be shorter if ``n`` is
        not a multiple of ``window_size``.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    return [slice(i, min(i + window_size, n)) for i in range(0, n, window_size)]


def assign_reference_per_window(
    hap: ArrayLike, refs: ArrayLike, window_size: int,
    max_mismatch_frac: float,
) -> np.ndarray:
    """
    For each window, decide which reference ``hap`` matches.

    This is analogous in spirit to gnomix's ``get_ref_map``, which tracks which
    of two reference haplotypes a given haplotype is following.

    The references can be allele-coded or ancestry-coded; only the pattern of
    matches/mismatches is used.

    Parameters
    ----------
    hap : (L,) array_like of int
        Haplotype of interest.
    refs : (R, L) array_like of int
        Reference haplotypes. Each row ``refs[r]`` is a reference pattern
        (allele-coded or ancestry-coded) of length L.
    window_size : int
        Number of SNPs per phasing window.
    max_mismatch_frac : float
        If both references mismatch more than this fraction of sites in a
        window, that window is treated as uninformative and assigned 0.

    Returns
    -------
    ref_track : (W,) np.ndarray of int8
        For each window ``w``:

        * 0 : ambiguous / low-confidence
        * 1..R : index (1-based) of the best-matching reference row
    """
    hap = np.asarray(hap)
    refs = np.asarray(refs)

    if refs.ndim != 2:
        raise ValueError("refs must be 2D with shape (n_ref, L).")

    n_ref, L = refs.shape
    if hap.shape[0] != L:
        raise ValueError("hap and refs must have the same length.")

    wslices = window_slices(L, window_size)
    ref_track = np.zeros(len(wslices), dtype=np.int8)

    for w_idx, sl in enumerate(wslices):
        h_win = hap[sl]
        mismatches = np.full(n_ref, np.inf, dtype=float)

        for r in range(n_ref):
            r_win = refs[r, sl]
            mask_valid = (r_win >= 0)
            if not np.any(mask_valid):
                continue
            mismatches[r] = np.mean(h_win[mask_valid] != r_win[mask_valid])

        best_r = int(np.argmin(mismatches))
        best_mism = mismatches[best_r]

        # If no reference had any valid sites or all mismatch too much
        if (not np.isfinite(best_mism)) or (best_mism >= max_mismatch_frac):
            ref_track[w_idx] = 0
        else:
            # 1-based index for compatibility with build_phase_track_from_ref
            ref_track[w_idx] = best_r + 1

    return ref_track

# ============================================================================
# Phase track and tail flipping
# ============================================================================

def build_phase_track_from_ref(ref_track: np.ndarray) -> np.ndarray:
    """
    Build a window-level "phase flip track" from reference assignments.

    The idea is: if the reference assignment for a haplotype changes
    frequently (e.g. 1 -> 2 or 2 -> 1), that indicates a possible phase flip
    between the two haplotypes.

    We construct a cumulative 0/1 track where each change in reference
    assignment toggles the phase. Ambiguous windows (label 0) simply inherit
    the current phase state.

    Parameters
    ----------
    ref_track : (W,) array_like of int
        Window-level reference assignments: 0, 1, or 2.

    Returns
    -------
    phase_track : (W,) np.ndarray of int8
        0/1 flag per window. When this track changes (0 -> 1 or 1 -> 0),
        windows from that point onward should have M/P swapped.
    """
    ref_track = np.asarray(ref_track)
    W = ref_track.shape[0]

    phase_track = np.zeros(W, dtype=np.int8)

    last_ref: int = 0
    current_phase: int = 0

    for w in range(W):
        ref = int(ref_track[w])
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
    """Apply tail flips between ``hap0`` and ``hap1`` according to phase_track.

    This is analogous to gnomix's ``correct_phase_error``, which uses a
    0/1 "track" over windows to decide where to swap suffixes between the two
    haplotypes.

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
        Ancestry-coded haplotypes to be corrected.
    phase_track : (W,) array_like of int {0,1}
        0/1 flags per window; when this changes (0 -> 1 or 1 -> 0), tails are
        swapped from that SNP position onward.
    window_size : int
        Number of SNPs per window.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) np.ndarray of int
        Phase-corrected haplotypes.
    """
    hap0 = np.asarray(hap0).copy()
    hap1 = np.asarray(hap1).copy()
    phase_track = np.asarray(phase_track)

    W = phase_track.shape[0]

    # Identify window boundaries where phase state changes
    change_points = np.where(np.diff(phase_track) != 0)[0] + 1  # window indices
    # Convert window indices to SNP indices
    flip_positions = [int(w * window_size) for w in change_points if w < W]

    for pos in flip_positions:
        # swap tails from pos onward
        tmp = hap0[pos:].copy()
        hap0[pos:] = hap1[pos:]
        hap1[pos:] = tmp

    return hap0, hap1

# ============================================================================
# High-level orchestrator for one sample (local ancestry arrays)
# ============================================================================

def phase_local_ancestry_sample(
    hap0: ArrayLike, hap1: ArrayLike, refs: ArrayLike,
    config: Optional[PhasingConfig] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Perform gnomix-style phasing corrections for a single individual.

    The steps are:

    1. Identify heterozygous blocks where hap0 != hap1.
    2. For each block, compare hap0 against all R references in sliding
       windows using :func:`assign_reference_per_window_multi`.
    3. Build a phase track using :func:`build_phase_track_from_ref`.
    4. Apply tail flips within each block using :func:`apply_phase_track`.

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
        Two haplotypes for the individual (e.g. maternal/paternal) with
        ancestry labels at each locus.
    refs : (R, L) array_like of int
        Reference haplotypes. These might correspond to different ancestry
        groups (or multiple samples per group). Only their match/mismatch
        patterns are used.
    config : PhasingConfig, optional
        Configuration for window size, thresholds, etc.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) np.ndarray of int
        Phase-corrected haplotypes.
    """
    if config is None:
        config = PhasingConfig()

    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)
    refs = np.asarray(refs)

    if refs.ndim != 2:
        raise ValueError("refs must be 2D with shape (n_ref, L).")

    n_ref, L = refs.shape

    if hap0.shape != hap1.shape or hap0.shape[0] != L:
        raise ValueError("hap0, hap1, and refs must all have length L.")

    if L == 0:
        return hap0.copy(), hap1.copy()

    het_blocks = find_heterozygous_blocks(
        hap0, hap1, min_block_len=config.min_block_len
    )

    if config.verbose:
        print(f"[phase_local_ancestry_sample] "
              f"{len(het_blocks)} heterozygous blocks")

    hap0_corr = hap0.copy()
    hap1_corr = hap1.copy()

    for block_idx, block in enumerate(het_blocks):
        start, end = block.start, block.stop

        if config.verbose:
            print(f"  - block {block_idx}: {start}..{end} (len={end-start})")

        # Extract block
        h0_blk = hap0_corr[start:end]
        h1_blk = hap1_corr[start:end]
        refs_blk = refs[:, start:end]  # (R, block_len)

        # For simplicity, build reference assignment based on one haplotype
        ref_track = assign_reference_per_window(
            hap=h0_blk, refs=refs_blk,
            window_size=config.window_size,
            max_mismatch_frac=config.max_mismatch_frac,
        )

        phase_track = build_phase_track_from_ref(ref_track)

        # Apply flips inside this block only
        local_L = end - start
        local_W = int(np.ceil(local_L / config.window_size))

        # Ensure phase_track has the correct number of windows
        if phase_track.shape[0] != local_W:
            if phase_track.shape[0] > local_W:
                phase_track = phase_track[:local_W]
            else:
                pad_val = int(phase_track[-1]) if phase_track.size > 0 else 0
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

# ============================================================================
# Annotation utilities and building reference haplotypes from VCF
# ============================================================================

def load_sample_annotations(
    annot_path: str, sep: str = r"\s+", col_sample: str = "sample_id",
    col_group: str = "group",
) -> pd.DataFrame:
    """
    Load sample annotation file mapping sample_id -> group (e.g., ancestry).

    Expected format by default: two columns (no header)::

        sample_id   group

    Parameters
    ----------
    annot_path : str
        Path to text file with sample annotations.
    sep : str, default r"\s+"
        Field separator pattern (passed to :func:`pandas.read_csv`).
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
    annot: pd.DataFrame, group0: Optional[str] = None, group1: Optional[str] = None,
    col_sample: str = "sample_id", col_group: str = "group",
) -> Tuple[str, str]:
    """
    Choose two reference sample IDs from annotation table.

    Parameters
    ----------
    annot : pandas.DataFrame
        Output of :func:`load_sample_annotations`.
    group0, group1 : str, optional
        Group labels to choose ref0 and ref1 from.
        If ``group1`` is None, both references are drawn from ``group0``.
        If ``group0`` is None, uses the most frequent group in ``annot``.
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
        raise ValueError(
            "ref0 and ref1 ended up being the same sample; "
            "provide distinct groups or more samples."
        )

    return ref0_id, ref1_id


def build_reference_haplotypes_from_vcf(
    vcf_path: str, annot_path: str, chrom: str, positions: np.ndarray,
    groups: Optional[list[str]] = None, hap_index_in_vcf: int = 0,
    col_sample: str = "sample_id", col_group: str = "group",
) -> Tuple[np.ndarray, list[str]]:
    """
    Build multiple reference haplotypes (refs) from an indexed VCF, one
    hapolotype per ancestry (or group) label.

    The reference haplotypes are haploid sequences (one allele per site),
    which are then used as ``refs`` in gnomix-style phase correction.

    Strategy
    --------
    1. Load sample annotations.
    2. Determine which groups to use:
       * If ``groups`` is None, use all unique values in ``col_group``.
       * Otherwise, restrict to the requested groups.
    3. For each group, pick one representative sample (first occurrence).
    4. Extract the chosen haploid allele (0/1/2/...) for each reference
       sample at the provided positions.

    Parameters
    ----------
    vcf_path : str
        Path to bgzipped, indexed VCF/BCF file (for cyvcf2).
    annot_path : str
        Path to sample annotation file (see :func:`load_sample_annotations`).
    chrom : str
        Chromosome name to extract (e.g., "1", "chr1").
    positions : array_like of int, shape (L,)
        Positions (1-based bp) for which we want reference alleles.
    groups : list of str, optional
        Group labels to use. If None, uses all unique groups from the
        annotation file.
    hap_index_in_vcf : int, default 0
        Which haploid allele to take (0 or 1) from the genotype.
    col_sample : str, default "sample_id"
        Column name for sample IDs in the annotation file.
    col_group : str, default "group"
        Column name for group labels in the annotation file.

    Returns
    -------
    refs : (R, L) np.ndarray of int8
        Haploid reference haplotypes as allele codes (0, 1, 2, or -1 for
        missing). Each row corresponds to one group in ``group_labels``.
    group_labels : list of str
        Group labels (in the same order as the first axis of ``refs``).
    """
    positions = np.asarray(positions, dtype=np.int64)
    L = positions.shape[0]

    annot = load_sample_annotations(annot_path, col_sample=col_sample,
                                    col_group=col_group)

    # Determine which groups to use
    if groups is None:
        group_labels = sorted(annot[col_group].unique().tolist())
    else:
        group_labels = list(groups)
        missing = set(group_labels) - set(annot[col_group].unique())
        if missing:
            raise ValueError(
                f"Requested groups not found in annotation: {sorted(missing)}"
            )

    # For each group, pick a representative sample (first in table)
    rep_samples: list[str] = []
    for g in group_labels:
        df_g = annot[annot[col_group] == g]
        if df_g.empty:
            raise ValueError(f"No samples found for group '{g}'.")
        rep_samples.append(df_g[col_sample].iloc[0])

    vcf = VCF(vcf_path)
    sample_to_idx = {sid: i for i, sid in enumerate(vcf.samples)}

    # Map rep_samples -> VCF indices
    rep_indices: list[int] = []
    for sid in rep_samples:
        if sid not in sample_to_idx:
            raise ValueError(f"Reference sample '{sid}' not found in VCF.")
        rep_indices.append(sample_to_idx[sid])

    n_ref = len(rep_indices)

    # Prepare refs: -1 means "missing / not found"
    refs = np.full((n_ref, L), -1, dtype=np.int8)

    # map position -> index in positions array
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}

    region_str = str(chrom)
    for variant in vcf(region_str):
        pos = int(variant.POS)
        if pos not in pos_to_idx:
            continue
        idx = pos_to_idx[pos]

        gts = variant.genotypes  # (n_samples, 3 or 4) -> [a1, a2, phased, ...]
        for r_idx, sample_idx_vcf in enumerate(rep_indices):
            gt = gts[sample_idx_vcf]
            allele = gt[hap_index_in_vcf] if gt[hap_index_in_vcf] >= 0 else -1
            refs[r_idx, idx] = allele

    return refs, group_labels

# ============================================================================
# Convenience wrapper: phase using VCF-derived references
# ============================================================================

def phase_local_ancestry_sample_from_vcf(
    hap0: np.ndarray, hap1: np.ndarray, positions: np.ndarray,
    chrom: str, ref_vcf_path: str, sample_annot_path: str,
    groups: Optional[list[str]] = None, 
    config: Optional[PhasingConfig] = None, hap_index_in_vcf: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-correct local ancestry haplotypes using VCF-derived references.

    This is a convenience wrapper that:

    1. Loads sample annotations to choose reference samples (ref0, ref1).
    2. Builds haploid reference haplotypes for those samples at the
       positions of interest using :mod:`cyvcf2`.
    3. Calls :func:`phase_local_ancestry_sample` with these references to
       perform gnomix-style tail-flip corrections.

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
        Local ancestry haplotypes for the individual (e.g. 0/1/2 for
        ancestry labels) on a given chromosome.
    positions : (L,) array_like of int
        1-based bp positions corresponding to hap0/hap1 entries.
    chrom : str
        Chromosome name to use when querying the VCF (e.g. "1" or "chr1").
    ref_vcf_path : str
        Path to bgzipped, indexed reference VCF/BCF file.
    sample_annot_path : str
        Path to sample annotation file (sample_id + group label).
    groups : list of str, optional
        Group labels to use. If None, uses all unique groups from the
        annotation file.
    config : PhasingConfig, optional
        Configuration for phasing (window size, thresholds, verbosity).
    hap_index_in_vcf : int, default 0
        Which haploid allele to take from the reference VCF genotypes.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) np.ndarray of int
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
    refs, groups = build_reference_haplotypes_from_vcf(
        vcf_path=ref_vcf_path, annot_path=sample_annot_path,
        chrom=chrom, positions=positions, groups=groups,
        hap_index_in_vcf=hap_index_in_vcf,
    )

    # Run gnomix-style phasing correction
    hap0_corr, hap1_corr = phase_local_ancestry_sample(
        hap0=hap0, hap1=hap1, refs=refs, config=config,
    )

    return hap0_corr, hap1_corr

# ============================================================================
# Metrics: switch error counting
# ============================================================================

def count_switch_errors(
    M_pred: ArrayLike, P_pred: ArrayLike, M_true: ArrayLike,
    P_true: ArrayLike,
) -> int:
    """
    Count minimal number of phase switches between predicted and truth.

    Counts the minimal number of suffix flips needed to turn
    ``(M_pred, P_pred)`` into ``(M_true, P_true)``. This is functionally
    similar to gnomix's ``find_switches``.

    Parameters
    ----------
    M_pred, P_pred : (L,) array_like of int
        Predicted ancestry haplotypes.
    M_true, P_true : (L,) array_like of int
        Ground-truth ancestry haplotypes.

    Returns
    -------
    n_switches : int
        Number of phase switch points.

    Raises
    ------
    RuntimeError
        If suffix swapping cannot transform the prediction into the truth.
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

# =============================================================================
# RFMix utilities: reconstructing haps & recombining counts
# =============================================================================

def build_hap_labels_from_rfmix(
    X_raw: DaskArray | np.ndarray, hap_index: np.ndarray, sample_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct hap0/hap1 ancestry labels for a single sample from RFMix.

    Parameters
    ----------
    X_raw : (L, n_cols) dask.array or np.ndarray
        Original RFMix matrix before summing haps. Columns correspond to
        (ancestry, hap, sample) combinations.
    hap_index : (n_samples, n_anc, 2) np.ndarray
        Output from RFMix's population subsetting. ``hap_index[s, a, 0/1]``
        gives the column index in ``X_raw`` for hap0 / hap1 of sample ``s``
        and ancestry ``a``.
    sample_idx : int
        Index of the sample to reconstruct.

    Returns
    -------
    hap0_labels, hap1_labels : (L,) np.ndarray of int
        Per-locus ancestry labels (0..n_anc-1) for hap0 and hap1.
        ``-1`` indicates missing (no ancestry column > 0 at that locus).
    """
    hap_index = np.asarray(hap_index)
    hap_idx_s = hap_index[sample_idx]  # (n_anc, 2)
    cols_h0 = hap_idx_s[:, 0]
    cols_h1 = hap_idx_s[:, 1]

    # Extract only the relevant columns.
    if isinstance(X_raw, da.Array):
        H0 = X_raw[:, cols_h0].compute()
        H1 = X_raw[:, cols_h1].compute()
    else:
        X = np.asarray(X_raw)
        H0 = X[:, cols_h0]
        H1 = X[:, cols_h1]

    H0 = np.asarray(H0)
    H1 = np.asarray(H1)

    # Ancestry label per hap = argmax over ancestries
    hap0_labels = np.argmax(H0, axis=1).astype(np.int16)
    hap1_labels = np.argmax(H1, axis=1).astype(np.int16)

    # Mark loci that are all-zero (no assignment) as -1
    mask0 = (H0.sum(axis=1) == 0)
    mask1 = (H1.sum(axis=1) == 0)
    hap0_labels[mask0] = -1
    hap1_labels[mask1] = -1

    return hap0_labels, hap1_labels


def combine_haps_to_counts(
    hap0: np.ndarray, hap1: np.ndarray, n_anc: int,
) -> np.ndarray:
    """
    Combine haplotype-level ancestry labels back into summed counts (0,1,2).

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
        Per-locus ancestry labels for each haplotype (0..n_anc-1 or -1).
    n_anc : int
        Number of ancestry classes.

    Returns
    -------
    out : (L, n_anc) np.ndarray of int8
        Summed counts for each ancestry (0, 1, or 2 per locus). Missing
        labels (``-1``) are ignored.
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

# ============================================================================
# High-level: phase a single admixed sample using RFMix + VCF
# ============================================================================

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
    groups: Optional[list[str]] = None,
    hap_index_in_vcf: int = 0,
) -> np.ndarray:
    """
    Phase-correct local ancestry for one sample using RFMix + VCF.

    Steps
    -----
    1. Use ``X_raw`` + ``hap_index`` to build hap0/hap1 ancestry labels.
    2. Run gnomix-style phasing via :func:`phase_local_ancestry_sample_from_vcf`.
    3. Recombine corrected hap0/hap1 into 0/1/2 counts.

    Parameters
    ----------
    admix_sample : (L, A) np.ndarray of int
        Summed local ancestry (0/1/2) for this sample; used for L and A.
    X_raw : (L, n_cols) dask.array or np.ndarray
        Original RFMix hap-by-ancestry matrix.
    hap_index : (n_samples, A, 2) np.ndarray
        Column mapping output by RFMix's population subsetting.
    sample_idx : int
        Sample index (0..n_samples-1).
    positions : (L,) array_like of int
        1-based genomic positions.
    chrom : str
        Chromosome name.
    ref_vcf_path : str
        Path to bgzipped, indexed reference VCF/BCF file.
    sample_annot_path : str
        Path to sample annotation file (sample_id + group label).
    config : PhasingConfig
        Phasing configuration.
    groups : list of str, optional
        Group labels to use. If None, uses all unique groups from the
        annotation file.
    hap_index_in_vcf : int, default 0
        Which haploid allele to take from the reference VCF (0 or 1).

    Returns
    -------
    admix_corr : (L, A) np.ndarray of int
        Phase-corrected summed ancestry counts for this sample.
    """
    admix_sample = np.asarray(admix_sample)
    positions = np.asarray(positions, dtype=np.int64)

    L, A = admix_sample.shape

    # Reconstruct hap0/hap1 ancestry labels from raw RFMix output
    hap0, hap1 = build_hap_labels_from_rfmix(
        X_raw=X_raw, hap_index=hap_index, sample_idx=sample_idx,
    )

    if hap0.shape[0] != L:
        raise ValueError("Length of hap0/hap1 does not match admix_sample.")

    # Gnomix-style phasing using reference VCF
    hap0_corr, hap1_corr = phase_local_ancestry_sample_from_vcf(
        hap0=hap0, hap1=hap1, positions=positions, chrom=chrom,
        ref_vcf_path=ref_vcf_path, sample_annot_path=sample_annot_path,
        groups=groups, config=config, hap_index_in_vcf=hap_index_in_vcf,
    )

    # Recombine corrected haplotypes to summed counts
    admix_corr = combine_haps_to_counts(hap0_corr, hap1_corr, n_anc=A)
    return admix_corr

# ============================================================================
# High-level: phase all samples in a Dask (L, S, A) array
# ============================================================================

def phase_admix_dask_with_index(
    admix: DaskArray,                # (L, S, A) summed counts
    X_raw: DaskArray | np.ndarray,   # (L, n_cols) raw RFMix matrix
    hap_index: np.ndarray,           # (S, A, 2)
    positions: np.ndarray,           # (L,)
    chrom: str,
    ref_vcf_path: str,
    sample_annot_path: str,
    config: PhasingConfig,
    groups: Optional[list[str]] = None,
    hap_index_in_vcf: int = 0,
) -> DaskArray:
    """
    Phase-correct all samples in a ``(L, S, A)`` admix Dask array.

    This function builds a Dask graph with one delayed phasing task per sample.
    Each task:

    1. Materializes the sample's admixture vector ``admix[:, s, :]``.
    2. Calls :func:`phase_admix_sample_from_vcf_with_index`.

    Parameters
    ----------
    admix : (L, S, A) dask.array.Array
        Summed local ancestry counts (0,1,2) for all samples.
    X_raw : (L, n_cols) dask.array.Array or np.ndarray
        Original RFMix matrix.
    hap_index : (S, A, 2) np.ndarray
        Column mapping from sample/ancestry to RFMix columns.
    positions : (L,) array_like of int
        Genomic positions.
    chrom : str
        Chromosome name.
    ref_vcf_path : str
        Path to bgzipped, indexed reference VCF/BCF file.
    sample_annot_path : str
        Path to sample annotation file (sample_id + group label).
    config : PhasingConfig
        Phasing configuration.
    groups : list of str, optional
        Group labels to use. If None, uses all unique groups from the
        annotation file.
    hap_index_in_vcf : int, default 0
        Which haploid allele to take from the reference VCF (0 or 1).

    Returns
    -------
    admix_corr : (L, S, A) dask.array.Array
        Phase-corrected local ancestry counts.
    """
    if not isinstance(admix, da.Array):
        raise TypeError("admix must be a dask.array.Array")

    n_loci, n_samples, n_anc = admix.shape

    # Create one delayed phasing task per sample
    delayed_results = []
    for s in range(n_samples):
        admix_s = admix[:, s, :]  # (L, A) dask slice

        @dask.delayed
        def _phase_one_sample(admix_s_block: DaskArray, sample_idx: int) -> np.ndarray:
            admix_s_np = admix_s_block.compute()
            return phase_admix_sample_from_vcf_with_index(
                admix_sample=admix_s_np, X_raw=X_raw, hap_index=hap_index,
                sample_idx=sample_idx, positions=positions, chrom=chrom,
                ref_vcf_path=ref_vcf_path,
                sample_annot_path=sample_annot_path, config=config,
                groups=groups, hap_index_in_vcf=hap_index_in_vcf,
            )

        delayed_results.append(_phase_one_sample(admix_s, s))

    # delayed_results is a list of (L, A) arrays, one per sample
    stacked = dask.delayed(lambda arrs: np.stack(arrs, axis=1))(delayed_results)  # (L, S, A)

    # Build a Dask array from the delayed stacked result
    admix_corr = da.from_delayed(stacked, shape=(n_loci, n_samples, n_anc), dtype=np.int8)

    return admix_corr
