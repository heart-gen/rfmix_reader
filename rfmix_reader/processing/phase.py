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

Reference haplotypes are read from *VCF-Zarr* stores produced by
`vcf2zarr` (bio2zarr) or `sgkit.io.vcf.vcf_reader.vcf_to_zarr`, which
follow the VCF-Zarr spec:

    - coords:  variant_position, sample_id
    - data:    call_genotype (variants, samples, ploidy)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import xarray as xr
import dask.array as da
from dask.array import Array as DaskArray

try:  # Optional GPU support
    import cupy as cp
except Exception:  # pragma: no cover - cupy is optional and may not be installed
    cp = None

try:  # Optional GPU DataFrame support
    import cudf
except Exception:  # pragma: no cover - cudf is optional and may not be installed
    cudf = None

ArrayLike = np.ndarray
logger = logging.getLogger(__name__)


def _get_array_module(*arrays):
    """Return cupy or numpy based on the input arrays."""

    if cp is not None:
        for arr in arrays:
            if isinstance(arr, cp.ndarray):
                return cp
    return np


def _to_pandas_dataframe(obj):
    """Convert cudf.DataFrame to pandas for consistent downstream handling."""

    if cudf is not None and isinstance(obj, cudf.DataFrame):
        return obj.to_pandas()
    return obj


def _series_to_array(series):
    """Convert pandas or cuDF Series to a NumPy or CuPy array."""

    if cudf is not None and isinstance(series, cudf.Series):
        if hasattr(series, "to_cupy"):
            try:
                return series.to_cupy()
            except Exception:
                pass
        return series.to_numpy()
    return series.to_numpy()


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


def _find_heterozygous_blocks(
    hap0: ArrayLike, hap1: ArrayLike, min_block_len: int = 1, max_gap: int = 1
) -> List[slice]:
    """
    Find contiguous blocks where ``hap0 != hap1`` (heterozygous ancestry).

    This follows the same conceptual idea as gnomix's ``_find_hetero_regions``,
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
    max_gap : int, default 1
        Maximum length of homozygous gap allowed when merging adjacent
        heterozygous runs. If two heterozygous segments are separated by a
        short homozygous stretch (``<= max_gap``), they are merged and the
        combined span is used to evaluate ``min_block_len``.

    Returns
    -------
    blocks : list of slice
        List of index slices (start:end) for heterozygous regions.
    """
    xp = _get_array_module(hap0, hap1)

    hap0 = xp.asarray(hap0)
    hap1 = xp.asarray(hap1)

    if hap0.shape != hap1.shape:
        raise ValueError("hap0 and hap1 must have the same shape.")

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative.")

    het = hap0 != hap1
    if not np.any(het):
        return []

    boundaries = xp.concatenate(
        ([0], xp.where(het[:-1] != het[1:])[0] + 1, [len(het)])
    )

    runs: List[Tuple[int, int]] = []
    for b in range(len(boundaries) - 1):
        start, end = boundaries[b], boundaries[b + 1]
        if het[start]:
            runs.append((start, end))

    if not runs:
        return []

    merged: List[slice] = []
    cur_start, cur_end = runs[0]

    for next_start, next_end in runs[1:]:
        gap = next_start - cur_end
        if gap <= max_gap:
            cur_end = next_end
        else:
            if (cur_end - cur_start) >= min_block_len:
                merged.append(slice(cur_start, cur_end))
            cur_start, cur_end = next_start, next_end

    if (cur_end - cur_start) >= min_block_len:
        merged.append(slice(cur_start, cur_end))

    return merged


def _window_slices(n: int, window_size: int) -> List[slice]:
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


def _assign_reference_per_window(
    hap: ArrayLike, refs: ArrayLike, window_size: int, max_mismatch_frac: float,
) -> np.ndarray:
    """
    For each window, decide which reference ``hap`` matches.

    Conceptually similar to gnomix's ``get_ref_map`` – we track which of the
    reference haplotypes a given haplotype is following.

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
    xp = _get_array_module(hap, refs)

    hap = xp.asarray(hap)
    refs = xp.asarray(refs)

    if refs.ndim != 2:
        raise ValueError("refs must be 2D with shape (n_ref, L).")

    n_ref, L = refs.shape
    if hap.shape[0] != L:
        raise ValueError("hap and refs must have the same length.")

    wslices = _window_slices(L, window_size)
    ref_track = xp.zeros(len(wslices), dtype=np.int8)

    for w_idx, sl in enumerate(wslices):
        h_win = hap[sl]
        r_win = refs[:, sl]
        mask_valid = r_win >= 0

        valid_counts = mask_valid.sum(axis=1)
        mismatch_counts = xp.sum((r_win != h_win) & mask_valid, axis=1)

        mismatches = xp.full(n_ref, xp.inf, dtype=float)
        xp.divide(
            mismatch_counts,
            valid_counts,
            out=mismatches,
            where=valid_counts > 0,
        )

        best_r = int(xp.argmin(mismatches))
        best_mism = mismatches[best_r]

        # Check for ties among references
        ties = xp.isclose(mismatches, best_mism)
        n_ties = ties.sum()

        # If tie across multiple references → ambiguous
        if n_ties > 1:
            ref_track[w_idx] = 0
            continue

        # if mismatch exceeds threshold → ambiguous
        if (not np.isfinite(best_mism)) or (best_mism >= max_mismatch_frac):
            ref_track[w_idx] = 0
        else:
            ref_track[w_idx] = best_r + 1  # 1-based

    return ref_track


def _build_phase_track_from_ref(ref_track: np.ndarray) -> np.ndarray:
    """
    Build a window-level "phase flip track" from reference assignments.

    When the reference assignment changes (1 -> 2 or 2 -> 1), that signals a
    possible phase flip. We build a cumulative 0/1 track where each change
    toggles the phase.

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
    xp = _get_array_module(ref_track)

    ref_track = xp.asarray(ref_track)
    W = ref_track.shape[0]

    phase_track = xp.zeros(W, dtype=np.int8)

    last_ref: int = 0
    current_phase: int = 0

    for w in range(W):
        ref = int(ref_track[w])
        if ref in (1, 2):
            if last_ref == 0:
                last_ref = ref
            elif ref != last_ref:
                current_phase ^= 1
                last_ref = ref
        phase_track[w] = current_phase

    return phase_track


def _apply_phase_track(
    hap0: ArrayLike, hap1: ArrayLike, phase_track: np.ndarray, window_size: int,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Apply tail flips between ``hap0`` and ``hap1`` according to phase_track.

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
    xp = _get_array_module(hap0, hap1, phase_track)

    hap0 = xp.asarray(hap0).copy()
    hap1 = xp.asarray(hap1).copy()
    phase_track = xp.asarray(phase_track)

    change_points = xp.where(xp.diff(phase_track) != 0)[0] + 1

    if change_points.size:
        flip_positions = change_points * window_size

        flip_mask = xp.zeros(hap0.shape[0] + 1, dtype=bool)
        flip_mask[flip_positions] ^= True
        flip_mask = xp.logical_xor.accumulate(flip_mask)[:-1]

        tmp = hap0[flip_mask].copy()
        hap0[flip_mask] = hap1[flip_mask]
        hap1[flip_mask] = tmp

    return hap0, hap1


def phase_local_ancestry_sample(
    hap0: ArrayLike, hap1: ArrayLike, refs: ArrayLike,
    config: Optional[PhasingConfig] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Perform gnomix-style phasing corrections for a single individual.

    Steps
    -----
    1. Identify heterozygous blocks where hap0 != hap1.
    2. For each block, compare hap0 against references in windows.
    3. Build a phase track from reference changes.
    4. Apply tail flips within each block.

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
        Two haplotypes for the individual with ancestry labels at each locus.
    refs : (R, L) array_like of int
        Reference haplotypes.
    config : PhasingConfig, optional
        Configuration for window size, thresholds, etc.

    Returns
    -------
    hap0_corr, hap1_corr : (L,) np.ndarray of int
        Phase-corrected haplotypes.
    """
    if config is None:
        config = PhasingConfig()

    xp = _get_array_module(hap0, hap1, refs)

    hap0 = xp.asarray(hap0)
    hap1 = xp.asarray(hap1)
    refs = xp.asarray(refs)

    if refs.ndim != 2:
        raise ValueError("refs must be 2D with shape (n_ref, L).")

    n_ref, L = refs.shape
    if hap0.shape != hap1.shape or hap0.shape[0] != L:
        raise ValueError("hap0, hap1, and refs must all have length L.")

    if L == 0:
        return hap0.copy(), hap1.copy()

    het_blocks = _find_heterozygous_blocks(
        hap0, hap1, min_block_len=config.min_block_len
    )

    if config.verbose:
        logger.info(
            "[phase_local_ancestry_sample] %d heterozygous blocks",
            len(het_blocks),
        )

    hap0_corr = hap0.copy()
    hap1_corr = hap1.copy()

    for block_idx, block in enumerate(het_blocks):
        start, end = block.start, block.stop

        if config.verbose:
            logger.info(
                "  - block %d: %d..%d (len=%d)",
                block_idx,
                start,
                end,
                end - start,
            )

        h0_blk = hap0_corr[start:end]
        h1_blk = hap1_corr[start:end]
        refs_blk = refs[:, start:end]

        ref_track = _assign_reference_per_window(
            hap=h0_blk,
            refs=refs_blk,
            window_size=config.window_size,
            max_mismatch_frac=config.max_mismatch_frac,
        )

        phase_track = _build_phase_track_from_ref(ref_track)

        local_L = end - start
        local_W = int(xp.ceil(local_L / config.window_size))

        if phase_track.shape[0] != local_W:
            if phase_track.shape[0] > local_W:
                phase_track = phase_track[:local_W]
            else:
                pad_val = int(phase_track[-1]) if phase_track.size > 0 else 0
                phase_track = xp.pad(
                    phase_track,
                    (0, local_W - phase_track.shape[0]),
                    constant_values=pad_val,
                )

        h0_blk_corr, h1_blk_corr = _apply_phase_track(
            h0_blk, h1_blk, phase_track, config.window_size
        )

        hap0_corr[start:end] = h0_blk_corr
        hap1_corr[start:end] = h1_blk_corr

    return hap0_corr, hap1_corr


def _load_sample_annotations(
    annot_path: str, sep: str = r"\s+", col_sample: str = "sample_id",
    col_group: str = "group",
) -> pd.DataFrame:
    """
    Load sample annotation file mapping sample_id -> group (e.g., ancestry).

    Expected default format: two columns (no header)::

        sample_id   group

    Parameters
    ----------
    annot_path : str
    sep : str
    col_sample : str
    col_group : str

    Returns
    -------
    annot : pandas.DataFrame
    """
    annot = pd.read_csv(
        annot_path, sep=sep, header=None,
        names=[col_sample, col_group],
        dtype={col_sample: str, col_group: str},
    )
    return annot


def _resolve_chrom_zarr_store(zarr_root: str, chrom: str) -> Path:
    """
    Find the VCF-Zarr store for a chromosome or raise with guidance.

    Rules
    -----
    - If ``zarr_root`` is a ``*.zarr`` path and exists, use it.
    - Otherwise search within ``zarr_root`` for:
        <chrom>.zarr, chr<chrom>.zarr, <chrom>, chr<chrom>
    """
    root = Path(zarr_root)

    if root.suffix == ".zarr":
        if root.exists():
            return root
        raise FileNotFoundError(
            f"Reference Zarr store not found: '{root}'.\n"
            "Generate it with convert_vcf_to_zarr / convert_vcfs_to_zarr "
            "(or `python -m rfmix_reader.cli.prepare_reference`)."
        )

    chrom_clean = chrom.removeprefix("chr")
    candidates = []
    for label in {chrom, f"chr{chrom_clean}", chrom_clean}:
        candidates.append(root / f"{label}.zarr")
        candidates.append(root / label)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No Zarr store found for chromosome '{chrom}' under '{zarr_root}'.\n"
        "Run convert_vcf_to_zarr / convert_vcfs_to_zarr or the CLI "
        "(`python -m rfmix_reader.cli.prepare_reference`) to create it."
    )


def build_reference_haplotypes_from_zarr(
    zarr_root: str, annot_path: str, chrom: str, positions: np.ndarray,
    groups: Optional[list[str]] = None, hap_index_in_zarr: int = 0,
    col_sample: str = "sample_id", col_group: str = "group",
    missing_loci_threshold: float = 0.05, raise_on_missing: bool = False,
) -> Tuple[np.ndarray, list[str], Dict[str, object]]:
    """
    Build reference haplotypes directly from a chromosome-specific VCF-Zarr store.

    Parameters
    ----------
    zarr_root : str
        Path to a ``*.zarr`` store for the chromosome or a directory containing
        per-chromosome Zarr stores.
    annot_path : str
        Path to sample annotation file (see :func:`_load_sample_annotations`).
    chrom : str
        Chromosome name to extract (e.g., "1", "chr1").
    positions : array_like of int, shape (L,)
        1-based bp positions for which we want reference alleles.
    groups : list of str, optional
        Group labels to use. If None, uses all unique groups.
    hap_index_in_zarr : int, default 0
        Which haploid allele to take (0 or 1) from the ploidy axis.
    col_sample : str, default "sample_id"
    col_group : str, default "group"
    missing_loci_threshold : float, default 0.05
        Max allowed fraction of requested loci that may be missing.
    raise_on_missing : bool, default False
        If True, raise when missing fraction exceeds threshold; otherwise log.

    Returns
    -------
    refs : (R, L) np.ndarray of int8
        Haploid reference haplotypes as allele codes (0, 1, 2, or -1 for
        missing). Each row corresponds to one group in ``group_labels``.
    group_labels : list[str]
        Group labels (same order as refs axis 0).
    match_stats : dict
        Keys: total_requested, matched_count, matched_fraction,
              missing_count, missing_fraction, missing_loci
    """
    if hasattr(positions, "to_numpy"):
        positions = positions.to_numpy()
    positions = np.asarray(positions, dtype=np.int64)
    L = positions.shape[0]

    annot = _load_sample_annotations(
        annot_path, col_sample=col_sample, col_group=col_group
    )

    if groups is None:
        group_labels = sorted(annot[col_group].unique().tolist())
    else:
        group_labels = list(groups)
        missing_groups = set(group_labels) - set(annot[col_group].unique())
        if missing_groups:
            raise ValueError(
                f"Requested groups not found in annotation: "
                f"{sorted(missing_groups)}"
            )

    rep_samples: list[str] = []
    for g in group_labels:
        df_g = annot[annot[col_group] == g]
        if df_g.empty:
            raise ValueError(f"No samples found for group '{g}'.")
        rep_samples.append(df_g[col_sample].iloc[0])

    zarr_path = _resolve_chrom_zarr_store(zarr_root, chrom)
    ds = xr.open_zarr(zarr_path)

    # Dimension names in VCF-Zarr: variants, samples, ploidy
    # Coordinates: sample_id, variant_position
    if "sample_id" not in ds:
        raise KeyError(
            "Zarr store missing 'sample_id' coordinate. "
            "Ensure it was written with vcf2zarr / vcf_to_zarr."
        )
    sample_to_idx = {
        sid: i for i, sid in enumerate(ds["sample_id"].values.astype(str))
    }
    missing_rep = [s for s in rep_samples if s not in sample_to_idx]
    if missing_rep:
        missing_fmt = ", ".join(missing_rep)
        raise ValueError(
            "Representative samples not found in Zarr store: "
            f"{missing_fmt}. "
            "Regenerate the Zarr store or update the sample annotations to match."
        )

    rep_indices = [sample_to_idx[s] for s in rep_samples]
    n_ref = len(rep_indices)

    # Positions: prefer 'variant_position' (VCF-Zarr spec).
    if "variant_position" in ds:
        variant_pos = np.asarray(ds["variant_position"].values, dtype=np.int64)
    elif "variants/POS" in ds:
        variant_pos = np.asarray(ds["variants/POS"].values, dtype=np.int64)
    else:
        raise KeyError(
            "Zarr store missing 'variant_position' (or 'variants/POS'). "
            "This does not look like a VCF-Zarr store."
        )

    pos_to_zarr_idx = {int(p): i for i, p in enumerate(variant_pos)}

    sort_idx = np.argsort(positions)
    positions_sorted = positions[sort_idx]

    refs_sorted = np.full((n_ref, L), -1, dtype=np.int8)

    matched_zarr_indices: list[int] = []
    matched_ref_positions: list[int] = []
    for i, pos in enumerate(positions_sorted):
        zidx = pos_to_zarr_idx.get(int(pos))
        if zidx is not None:
            matched_zarr_indices.append(zidx)
            matched_ref_positions.append(i)

    matched_count = len(matched_zarr_indices)
    missing_count = L - matched_count
    matched_fraction = matched_count / L if L > 0 else 0.0
    missing_fraction = missing_count / L if L > 0 else 0.0

    match_stats: Dict[str, object] = {
        "total_requested": int(L),
        "matched_count": int(matched_count),
        "matched_fraction": float(matched_fraction),
        "missing_count": int(missing_count),
        "missing_fraction": float(missing_fraction),
        "missing_loci": positions_sorted[
            [i for i in range(L) if i not in set(matched_ref_positions)]
        ]
        if missing_count > 0
        else np.array([], dtype=np.int64),
    }

    if missing_fraction > missing_loci_threshold:
        msg = (
            f"Reference Zarr store for chrom={chrom} is missing "
            f"{missing_fraction:.1%} of requested loci "
            f"({missing_count}/{L})."
        )
        if raise_on_missing:
            raise ValueError(msg)
        logger.warning(msg)

    if matched_zarr_indices:
        matched_zarr_indices_arr = np.asarray(
            matched_zarr_indices, dtype=np.int64
        )
        matched_ref_positions_arr = np.asarray(
            matched_ref_positions, dtype=np.int64
        )

        if "call_genotype" not in ds:
            raise KeyError(
                "Zarr store missing 'call_genotype'. "
                "Ensure it was written following the VCF-Zarr spec."
            )

        geno = ds["call_genotype"].isel(
            variants=matched_zarr_indices_arr,
            samples=rep_indices,
            ploidy=hap_index_in_zarr,
        )
        geno_data = geno.data
        if hasattr(geno_data, "compute"):
            geno_data = geno_data.compute()
        geno_arr = np.asarray(geno_data)
        geno_arr = np.where(geno_arr >= 0, geno_arr, -1).astype(np.int8)

        refs_sorted[:, matched_ref_positions_arr] = geno_arr.T

    refs = np.empty_like(refs_sorted)
    refs[:, sort_idx] = refs_sorted

    return refs, group_labels, match_stats


def phase_local_ancestry_sample_from_zarr(
    hap0: np.ndarray, hap1: np.ndarray, positions: np.ndarray, chrom: str,
    ref_zarr_root: str, sample_annot_path: str, groups: Optional[list[str]] = None,
    config: Optional[PhasingConfig] = None, hap_index_in_zarr: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Phase-correct local ancestry haplotypes using VCF-Zarr-derived references.

    Steps
    -----
    1. Load sample annotations and choose representative samples per group.
    2. Build haploid reference haplotypes from a chromosome-specific VCF-Zarr
       store at the target positions.
    3. Run gnomix-style tail-flip phasing.

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
    positions : (L,) array_like of int
    chrom : str
    ref_zarr_root : str
    sample_annot_path : str
    groups : list of str, optional
    config : PhasingConfig, optional
    hap_index_in_zarr : int, default 0

    Returns
    -------
    hap0_corr, hap1_corr : (L,) np.ndarray of int
        Phase-corrected local ancestry haplotypes.
    match_stats : dict
        Diagnostics about matched/missing loci in the reference.
    """
    if config is None:
        config = PhasingConfig()

    positions = np.asarray(positions, dtype=np.int64)
    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)

    if hap0.shape != hap1.shape or hap0.shape[0] != positions.shape[0]:
        raise ValueError("hap0, hap1, and positions must all have length L.")

    refs, group_labels, match_stats = build_reference_haplotypes_from_zarr(
        zarr_root=ref_zarr_root,
        annot_path=sample_annot_path,
        chrom=chrom,
        positions=positions,
        groups=groups,
        hap_index_in_zarr=hap_index_in_zarr,
    )

    if config.verbose:
        logger.info(
            "[phase_local_ancestry_sample_from_zarr] Using groups: %s",
            ", ".join(group_labels),
        )

    hap0_corr, hap1_corr = phase_local_ancestry_sample(
        hap0=hap0,
        hap1=hap1,
        refs=refs,
        config=config,
    )

    return hap0_corr, hap1_corr, match_stats


def count_switch_errors(
    M_pred: ArrayLike, P_pred: ArrayLike, M_true: ArrayLike,
    P_true: ArrayLike,
) -> int:
    """
    Count minimal number of phase switches between predicted and truth.

    Counts the minimal number of suffix flips needed to turn
    ``(M_pred, P_pred)`` into ``(M_true, P_true)``.
    """
    xp = _get_array_module(M_pred, P_pred, M_true, P_true)

    M_pred = xp.asarray(M_pred).copy()
    P_pred = xp.asarray(P_pred).copy()
    M_true = xp.asarray(M_true)
    P_true = xp.asarray(P_true)

    if not (M_pred.shape == P_pred.shape == M_true.shape == P_true.shape):
        raise ValueError("All haplotypes must have the same shape.")

    n_switches = 0
    L = M_pred.shape[0]
    for i in range(L):
        if M_pred[i] != M_true[i]:
            M_tmp = M_pred[i:].copy()
            M_pred[i:] = P_pred[i:]
            P_pred[i:] = M_tmp
            n_switches += 1

    if not (np.array_equal(M_pred, M_true) and np.array_equal(P_pred, P_true)):
        raise RuntimeError("Phase error correction did not align with truth.")

    return n_switches


def _build_hap_labels_from_rfmix(
    X_raw: DaskArray | np.ndarray, sample_idx: int, n_anc: int, n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct hap0/hap1 ancestry labels for a sample from the raw RFMix fb matrix.

    Parameters
    ----------
    X_raw : (L, n_cols) dask.array or np.ndarray
        Original RFMix matrix before summing haps. Columns correspond to
        (ancestry, hap, sample) combinations.
    sample_idx : int
    n_anc : int
    n_samples : int

    Returns
    -------
    hap0_labels, hap1_labels : (L,) np.ndarray of int
        Per-locus ancestry labels (0..n_anc-1) for hap0 and hap1.
        ``-1`` indicates missing (no ancestry column > 0 at that locus).
    """
    base = sample_idx * (n_anc * 2)
    cols_h0 = base + np.arange(n_anc) * 2
    cols_h1 = base + np.arange(n_anc) * 2 + 1

    if isinstance(X_raw, da.Array):
        H0 = X_raw[:, cols_h0].compute()
        H1 = X_raw[:, cols_h1].compute()
    else:
        xp = _get_array_module(X_raw)
        H0 = xp.asarray(X_raw)[:, cols_h0]
        H1 = xp.asarray(X_raw)[:, cols_h1]

    xp = _get_array_module(H0, H1)

    hap0_labels = xp.argmax(H0, axis=1).astype(np.int16)
    hap1_labels = xp.argmax(H1, axis=1).astype(np.int16)

    hap0_labels[H0.sum(axis=1) == 0] = -1
    hap1_labels[H1.sum(axis=1) == 0] = -1

    return hap0_labels, hap1_labels


def _combine_haps_to_counts(
    hap0: np.ndarray, hap1: np.ndarray, n_anc: int,
) -> np.ndarray:
    """
    Combine haplotype-level ancestry labels back into summed counts (0,1,2).

    Parameters
    ----------
    hap0, hap1 : (L,) array_like of int
    n_anc : int

    Returns
    -------
    out : (L, n_anc) np.ndarray of int8
    """
    xp = _get_array_module(hap0, hap1)

    hap0 = xp.asarray(hap0)
    hap1 = xp.asarray(hap1)

    if hap0.shape != hap1.shape:
        raise ValueError("hap0 and hap1 must have same shape.")

    L = hap0.shape[0]
    out = xp.zeros((L, n_anc), dtype=np.int8)

    idx = np.arange(L)

    valid0 = (hap0 >= 0) & (hap0 < n_anc)
    out[idx[valid0], hap0[valid0]] += 1

    valid1 = (hap1 >= 0) & (hap1 < n_anc)
    out[idx[valid1], hap1[valid1]] += 1

    return out


def phase_admix_sample_from_zarr_with_index(
    admix_sample: np.ndarray,  # (L, A) counts
    X_raw: DaskArray | np.ndarray,  # (L, n_cols) raw RFMix matrix
    sample_idx: int, n_samples: int, positions: np.ndarray,  # (L,)
    chrom: str, ref_zarr_root: str, sample_annot_path: str,
    config: PhasingConfig, groups: Optional[list[str]] = None,
    hap_index_in_zarr: int = 0, refs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Phase-correct local ancestry for one sample using RFMix + VCF-Zarr references.

    Steps
    -----
    1. Use ``X_raw`` to build hap0/hap1 ancestry labels.
    2. Run gnomix-style phasing via :func:`phase_local_ancestry_sample_from_zarr`.
    3. Recombine corrected hap0/hap1 into 0/1/2 counts.

    If ``refs`` is provided, precomputed reference haplotypes are reused and the
    Zarr loading step is skipped.

    Returns
    -------
    admix_corr : (L, A) np.ndarray of int
        Phase-corrected summed ancestry counts for this sample.
    """
    xp = _get_array_module(admix_sample, positions)

    admix_sample = xp.asarray(admix_sample)
    positions = xp.asarray(positions, dtype=xp.int64)

    L, A = admix_sample.shape

    hap0, hap1 = _build_hap_labels_from_rfmix(
        X_raw=X_raw, sample_idx=sample_idx, n_anc=A, n_samples=n_samples
    )

    if hap0.shape[0] != L:
        raise ValueError("Length of hap0/hap1 does not match admix_sample.")

    if refs is None:
        hap0_corr, hap1_corr, _match_stats = phase_local_ancestry_sample_from_zarr(
            hap0=hap0,
            hap1=hap1,
            positions=positions,
            chrom=chrom,
            ref_zarr_root=ref_zarr_root,
            sample_annot_path=sample_annot_path,
            groups=groups,
            config=config,
            hap_index_in_zarr=hap_index_in_zarr,
        )
    else:
        if refs.shape[1] != L:
            raise ValueError("Reference haplotypes do not match admix_sample length.")

        hap0_corr, hap1_corr = phase_local_ancestry_sample(
            hap0=hap0,
            hap1=hap1,
            refs=refs,
            config=config,
        )

    admix_corr = _combine_haps_to_counts(hap0_corr, hap1_corr, n_anc=A)
    return admix_corr


def phase_admix_dask_with_index(
    admix: DaskArray,  # (L, S, A) summed counts
    X_raw: DaskArray | np.ndarray,  # (L, n_cols) raw RFMix matrix
    positions: np.ndarray,  # (L,)
    chrom: str, ref_zarr_root: str, sample_annot_path: str,
    config: PhasingConfig, groups: Optional[list[str]] = None,
    hap_index_in_zarr: int = 0,
) -> DaskArray:
    """
    Phase-correct all samples in a ``(L, S, A)`` admix Dask array.

    Rechunks so each block covers one sample, then applies
    :func:`phase_admix_sample_from_zarr_with_index` via ``map_blocks``. Reference
    haplotypes are constructed once and reused across all samples.

    Returns
    -------
    admix_corr : (L, S, A) dask.array.Array
        Phase-corrected local ancestry counts.
    """
    if not isinstance(admix, da.Array):
        raise TypeError("admix must be a dask.array.Array")

    if isinstance(X_raw, da.Array):
        n_loci, n_cols = X_raw.shape
        X_raw = X_raw.rechunk((n_loci, min(64, n_cols)))

    n_loci, n_samples, n_anc = admix.shape
    admix_single_sample = admix.rechunk((n_loci, 1, n_anc))
    positions = np.asarray(positions, dtype=np.int64)

    refs, group_labels, _match_stats = build_reference_haplotypes_from_zarr(
        zarr_root=ref_zarr_root,
        annot_path=sample_annot_path,
        chrom=chrom,
        positions=positions,
        groups=groups,
        hap_index_in_zarr=hap_index_in_zarr,
    )

    if config.verbose:
        logger.info(
            "[phase_admix_dask_with_index] Using groups: %s",
            ", ".join(group_labels),
        )

    def _phase_block(admix_block: np.ndarray, block_info=None) -> np.ndarray:
        """Phase a single Dask block."""
        info = block_info[0]
        chunk_loc = info["chunk-location"]
        sample_block_idx = chunk_loc[1]

        block_sample_size = admix_block.shape[1]
        sample_start = sample_block_idx * block_sample_size
        sample_stop = sample_start + block_sample_size

        sample_indices = range(sample_start, sample_stop)
        xp_block = _get_array_module(admix_block)
        phased_block = xp_block.empty_like(admix_block, dtype=np.int8)
        for offset, sample_idx in enumerate(sample_indices):
            phased_block[:, offset, :] = phase_admix_sample_from_zarr_with_index(
                admix_sample=admix_block[:, offset, :], X_raw=X_raw,
                sample_idx=sample_idx, n_samples=n_samples, positions=positions,
                chrom=chrom, ref_zarr_root=ref_zarr_root,
                sample_annot_path=sample_annot_path, config=config, refs=refs,
                groups=groups, hap_index_in_zarr=hap_index_in_zarr,
            )

        return phased_block

    phased = da.map_blocks(
        _phase_block, admix_single_sample, dtype=np.int8,
        chunks=admix_single_sample.chunks,
    )

    return phased


def phase_rfmix_chromosome_to_zarr(
    file_prefix: str, ref_zarr_root: str, sample_annot_path: str,
    output_path: str, *, chrom: Optional[str] = None,
    groups: Optional[list[str]] = None,
    config: Optional[PhasingConfig] = None,
    hap_index_in_zarr: int = 0,
    binary_dir: str = "./binary_files",
    generate_binary: bool = False,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Phase local ancestry for a single chromosome and write to a Zarr store.

    This is a convenience wrapper around :func:`read_rfmix` and
    :func:`phase_admix_dask_with_index` that keeps processing per chromosome and
    serializes the phased local ancestry as an :class:`xarray.Dataset` with
    coordinates for positions, samples, ancestries, and chromosome labels.

    Parameters
    ----------
    file_prefix
        Prefix pointing to the RFMix outputs (``*.fb.tsv`` / ``*.rfmix.Q``).
    ref_zarr_root
        Path to a reference VCF-Zarr store (or directory of per-chromosome
        stores).
    sample_annot_path
        Two-column TSV mapping ``sample_id`` to reference group label.
    output_path
        Destination Zarr store to write.
    chrom
        Optional chromosome label to restrict which files are read.
    groups
        Optional subset of reference groups to include.
    config
        Optional :class:`PhasingConfig` instance controlling phasing
        parameters. Defaults to :class:`PhasingConfig()`.
    hap_index_in_zarr
        Which haplotype to read from the reference VCF-Zarr store (``0`` or
        ``1``).
    binary_dir
        Directory containing RFMix binary caches.
    generate_binary
        If :data:`True`, generate the binary caches prior to reading.
    verbose
        Control progress reporting when invoking :func:`read_rfmix`.

    Returns
    -------
    xarray.Dataset
        Dataset containing a ``local_ancestry`` variable with dimensions
        ``(variant, sample, ancestry)`` written to ``output_path``.
    """
    from ..readers.read_rfmix import read_rfmix

    config = config or PhasingConfig()

    loci_df, g_anc, admix, X_raw = read_rfmix(
        file_prefix,
        binary_dir=binary_dir,
        generate_binary=generate_binary,
        verbose=verbose,
        return_original=True,
        chrom=chrom,
    )

    loci_df = _to_pandas_dataframe(loci_df)
    g_anc = _to_pandas_dataframe(g_anc)

    chrom_labels = loci_df["chromosome"].astype(str).unique()
    if len(chrom_labels) != 1:
        raise ValueError(
            "Phasing per chromosome expects a single chromosome in the input."
        )

    chrom_label = str(chrom_labels[0])
    positions = _series_to_array(loci_df["physical_position"])

    pops = g_anc.drop(["sample_id", "chrom"], axis=1).columns.values
    sample_ids = g_anc["sample_id"].tolist()

    phased = phase_admix_dask_with_index(
        admix=admix,
        X_raw=X_raw,
        positions=positions,
        chrom=chrom_label,
        ref_zarr_root=ref_zarr_root,
        sample_annot_path=sample_annot_path,
        config=config,
        groups=groups,
        hap_index_in_zarr=hap_index_in_zarr,
    )

    dataset = xr.Dataset(
        {
            "local_ancestry": xr.DataArray(
                phased,
                dims=("variant", "sample", "ancestry"),
                coords={
                    "variant_position": ("variant", positions),
                    "chromosome": ("variant", loci_df["chromosome"].astype(str)),
                    "sample_id": ("sample", sample_ids),
                    "ancestry": ("ancestry", pops),
                },
                name="local_ancestry",
            )
        }
    )

    dataset.to_zarr(output_path, mode="w")
    return dataset


def merge_phased_zarrs(
    chrom_zarr_paths: List[str], output_path: str, *, sort: bool = True
) -> xr.Dataset:
    """
    Merge per-chromosome phased Zarr outputs along the variant axis.

    Parameters
    ----------
    chrom_zarr_paths
        List of paths to per-chromosome phased Zarr stores generated by
        :func:`phase_rfmix_chromosome_to_zarr`.
    output_path
        Destination Zarr store for the merged dataset.
    sort
        If :data:`True`, sort by ``chromosome`` then ``variant_position`` after
        concatenation.

    Returns
    -------
    xarray.Dataset
        Combined dataset written to ``output_path``.
    """
    datasets = [xr.open_zarr(Path(p)) for p in chrom_zarr_paths]

    if not datasets:
        raise ValueError("No Zarr paths provided for merging.")

    sample_ids = [tuple(ds.sample_id.values.tolist()) for ds in datasets]
    if len(set(sample_ids)) != 1:
        raise ValueError("Sample sets differ across per-chromosome Zarr stores.")

    ancestry_labels = [tuple(ds.ancestry.values.tolist()) for ds in datasets]
    if len(set(ancestry_labels)) != 1:
        raise ValueError("Ancestry labels differ across per-chromosome Zarr stores.")

    combined = xr.concat(datasets, dim="variant")
    if sort:
        combined = combined.sortby(["chromosome", "variant_position"])

    combined.to_zarr(output_path, mode="w")
    return combined
