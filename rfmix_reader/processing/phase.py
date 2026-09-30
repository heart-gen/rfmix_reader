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
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

import xarray as xr

from ..backends import _select_array_backend
from ..core.codes import to_str_array

if TYPE_CHECKING:
    from dask.array import Array as DaskArray

ArrayLike = np.ndarray
logger = logging.getLogger(__name__)


def _get_array_module(*arrays):
    """Return cupy or numpy based on the input arrays."""

    array_mod = _select_array_backend()
    if array_mod.__name__ == "cupy":
        for arr in arrays:
            if isinstance(arr, array_mod.ndarray):
                return array_mod
    return np


def _to_numpy_array(arr):
    """Convert CuPy arrays to NumPy; leave other inputs unchanged."""

    array_mod = _select_array_backend()
    if array_mod.__name__ == "cupy" and isinstance(arr, array_mod.ndarray):
        return array_mod.asnumpy(arr)
    return np.asarray(arr)


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
    posterior_margin : float, default 0.2
        gnomix method only.  A window whose mean orientation score (see
        :func:`gnomix_switch_mask`) has magnitude below this value is treated
        as uninformative and inherits the state of the previous window.
        Scores range from -2 (clearly swapped) to +2 (clearly in phase).
    verbose : bool, default False
        If True, prints basic diagnostics per sample / region.
    """
    window_size: int = 50
    min_block_len: int = 20
    max_mismatch_frac: float = 0.5
    posterior_margin: float = 0.2
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
    hap0 = _to_numpy_array(hap0)
    hap1 = _to_numpy_array(hap1)

    if hap0.shape != hap1.shape:
        raise ValueError("hap0 and hap1 must have the same shape.")

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative.")

    het = hap0 != hap1
    if not np.any(het):
        return []

    boundaries = np.concatenate(
        ([0], np.where(het[:-1] != het[1:])[0] + 1, [len(het)])
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
    hap0 = _to_numpy_array(hap0).copy()
    hap1 = _to_numpy_array(hap1).copy()
    phase_track = _to_numpy_array(phase_track)

    change_points = np.where(np.diff(phase_track) != 0)[0] + 1

    if change_points.size:
        flip_positions = change_points * window_size

        flip_mask = np.zeros(hap0.shape[0] + 1, dtype=bool)
        flip_mask[flip_positions] ^= True
        flip_mask = np.logical_xor.accumulate(flip_mask)[:-1]

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

    hap0 = _to_numpy_array(hap0)
    hap1 = _to_numpy_array(hap1)
    refs = _to_numpy_array(refs)

    xp = np

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
        sid: i for i, sid in enumerate(to_str_array(ds["sample_id"].values))
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


# Convenience aliases to satisfy existing tests; prefer the underscored
# implementations above for non-test usage.
find_heterozygous_blocks = _find_heterozygous_blocks
assign_reference_per_window = _assign_reference_per_window
build_phase_track_from_ref = _build_phase_track_from_ref
apply_phase_track = _apply_phase_track


def gnomix_switch_mask_sample(
    hap0: np.ndarray, hap1: np.ndarray, post0: Optional[np.ndarray],
    post1: Optional[np.ndarray], config: Optional[PhasingConfig] = None,
) -> np.ndarray:
    """
    gnomix-style switch-error detection for one sample (one chromosome).

    Parameters
    ----------
    hap0, hap1 : (L,) int
        Ancestry codes of the two haplotypes (``-1`` = missing).
    post0, post1 : (L, A) float or None
        Per-haplotype ancestry posteriors.  ``None`` uses one-hot codes (the
        hard-call variant).
    config : PhasingConfig

    Returns
    -------
    swapped : (L,) bool
        ``True`` where the two haplotypes are in the opposite orientation to
        the start of their heterozygous block and must be exchanged.

    Notes
    -----
    Heterozygous blocks are maximal runs where the two haplotypes carry
    different ancestries and the unordered pair ``{a, b}`` is constant; blocks
    shorter than ``config.min_block_len`` loci are left alone.  Within a block
    with start orientation ``(a, b)`` every locus gets the score
    ``(p0[a] + p1[b]) - (p0[b] + p1[a])``; the mean over windows of
    ``config.window_size`` loci is positive when the current orientation
    matches the block start and negative after a switch error.  Windows with
    ``|mean| < config.posterior_margin`` are uninformative and inherit the
    previous window's state.  Swapping every window in the "switched" state
    is equivalent to gnomix's successive tail flips at each change point.
    """
    config = config or PhasingConfig()
    hap0 = np.asarray(hap0).astype(np.int64)
    hap1 = np.asarray(hap1).astype(np.int64)
    L = hap0.shape[0]
    swapped = np.zeros(L, dtype=bool)
    if L == 0:
        return swapped

    valid = (hap0 >= 0) & (hap1 >= 0)
    het = valid & (hap0 != hap1)
    if not het.any():
        return swapped
    n_anc = int(max(hap0.max(), hap1.max())) + 1
    if post0 is None or post1 is None:
        eye = np.eye(n_anc, dtype=np.float32)
        post0 = eye[np.clip(hap0, 0, n_anc - 1)]
        post1 = eye[np.clip(hap1, 0, n_anc - 1)]
    post0 = np.asarray(post0, dtype=np.float32)
    post1 = np.asarray(post1, dtype=np.float32)
    n_anc = post0.shape[1]

    lo = np.minimum(hap0, hap1)
    hi = np.maximum(hap0, hap1)
    key = np.where(het, lo * n_anc + hi, -1)
    boundaries = np.flatnonzero(key[1:] != key[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [L]])

    W = int(config.window_size)
    for s0, s1 in zip(starts, stops):
        if key[s0] < 0 or (s1 - s0) < config.min_block_len:
            continue
        a, b = int(hap0[s0]), int(hap1[s0])
        score = (post0[s0:s1, a] + post1[s0:s1, b]) - (post0[s0:s1, b] + post1[s0:s1, a])
        n = s1 - s0
        n_win = -(-n // W)
        padded = np.full(n_win * W, np.nan, dtype=np.float32)
        padded[:n] = score
        means = np.nanmean(padded.reshape(n_win, W), axis=1)
        informative = np.abs(means) >= config.posterior_margin
        state = np.zeros(n_win, dtype=bool)
        current = False
        for w in range(n_win):
            if informative[w]:
                current = bool(means[w] < 0)
            state[w] = current
        swapped[s0:s1] = np.repeat(state, W)[:n]
    return swapped


def _swap_block(hap_block: np.ndarray, post_block: Optional[np.ndarray],
                config: PhasingConfig) -> np.ndarray:
    """map_blocks kernel: (L, s, 2) codes [+ (L, s, 2, A) posteriors] -> (L, s) bool."""
    hap_block = np.asarray(hap_block)
    out = np.zeros(hap_block.shape[:2], dtype=bool)
    for j in range(hap_block.shape[1]):
        p0 = p1 = None
        if post_block is not None:
            p0 = np.asarray(post_block[:, j, 0, :])
            p1 = np.asarray(post_block[:, j, 1, :])
        out[:, j] = gnomix_switch_mask_sample(hap_block[:, j, 0], hap_block[:, j, 1], p0, p1, config)
    return out


def gnomix_switch_mask(hap, posterior=None, *, config: Optional[PhasingConfig] = None) -> "DaskArray":
    """
    Lazy ``(L, S)`` boolean mask of loci whose haplotypes must be exchanged,
    computed per sample from ``(L, S, 2)`` codes and optional ``(L, S, 2, A)``
    posteriors with :func:`gnomix_switch_mask_sample`.
    """
    import dask.array as da

    config = config or PhasingConfig()
    hap = da.asarray(hap)
    L = hap.shape[0]
    hap1 = hap.rechunk((L, 1, 2))
    if posterior is None:
        return da.blockwise(
            _swap_block, "ls", hap1, "lsp", None, None, config, None,
            dtype=bool, concatenate=True,
        )
    post = da.asarray(posterior).rechunk((L, 1, 2, posterior.shape[3]))
    return da.blockwise(
        _swap_block, "ls", hap1, "lsp", post, "lspa", config, None,
        dtype=bool, concatenate=True,
    )


def _apply_swaps(hap, swapped, posterior=None):
    """Exchange the two haplotypes (codes and posteriors) where ``swapped``."""
    import dask.array as da

    hap = da.asarray(hap)
    # align the (one sample per block) mask with the data's chunking so the
    # result keeps the data's block structure
    swapped = da.asarray(swapped).rechunk((hap.chunks[0], hap.chunks[1]))
    codes = da.where(swapped[:, :, None], hap[:, :, ::-1], hap).astype(np.int8)
    if posterior is None:
        return codes, None
    post = da.asarray(posterior)
    post_c = da.where(swapped[:, :, None, None], post[:, :, ::-1, :], post).astype(np.float32)
    return codes, post_c


def phase_haplotypes(
    hap: "DaskArray | np.ndarray", positions: Optional[np.ndarray] = None,
    chrom: Optional[str] = None, ref_zarr_root: Optional[str] = None,
    sample_annot_path: Optional[str] = None, config: Optional[PhasingConfig] = None,
    groups: Optional[list[str]] = None, hap_index_in_zarr: int = 0,
    refs: Optional[np.ndarray] = None, *, method: str = "gnomix",
    posterior=None,
) -> "DaskArray":
    """
    Phase-correct ``(L, S, 2)`` haplotype ancestry codes for every sample.

    Parameters
    ----------
    method : {"gnomix", "reference"}
        ``"gnomix"`` (default) detects switch errors from the two haplotypes'
        own ancestry posteriors (``posterior``, ``(L, S, 2, A)``; one-hot codes
        when absent) — no reference panel needed.  ``"reference"`` is the
        previous implementation that matches ancestry labels against
        reference-panel *allele* codes (``ref_zarr_root``/``sample_annot_path``
        required); it is kept for comparison only, see the note.
    posterior : array, optional
        Per-haplotype posteriors for ``method="gnomix"``.

    .. note::
       The ``"reference"`` matcher compares ancestry labels against allele codes
       (see :func:`build_reference_haplotypes_from_zarr`), which is only
       meaningful when the two code spaces coincide.  Prefer ``"gnomix"``.

    Returns
    -------
    dask.array.Array, int8, ``(L, S, 2)``
    """
    import dask.array as da

    config = config or PhasingConfig()
    hap = da.asarray(hap)
    if hap.ndim != 3 or hap.shape[2] != 2:
        raise ValueError(f"hap must be (L, S, 2); got {hap.shape}.")
    L, n_samples, _ = hap.shape

    if method == "gnomix":
        swapped = gnomix_switch_mask(hap, posterior, config=config)
        codes, _ = _apply_swaps(hap, swapped, None)
        return codes
    if method != "reference":
        raise ValueError("method must be 'gnomix' or 'reference'.")

    if positions is None:
        raise ValueError("positions are required for method='reference'.")
    positions = np.asarray(positions, dtype=np.int64)
    if positions.shape[0] != L:
        raise ValueError("positions must have length L.")
    if refs is None:
        if ref_zarr_root is None or sample_annot_path is None or chrom is None:
            raise ValueError("ref_zarr_root, sample_annot_path and chrom are required "
                             "for method='reference'.")
        refs, group_labels, _stats = build_reference_haplotypes_from_zarr(
            zarr_root=ref_zarr_root, annot_path=sample_annot_path, chrom=chrom,
            positions=positions, groups=groups, hap_index_in_zarr=hap_index_in_zarr,
        )
        if config.verbose:
            logger.info("[phase_haplotypes] Using groups: %s", ", ".join(group_labels))
    refs = _to_numpy_array(refs)
    if refs.shape[1] != L:
        raise ValueError("Reference haplotypes do not match the number of loci.")

    def _phase_block(block: np.ndarray) -> np.ndarray:
        block = np.asarray(block)
        out = np.empty_like(block, dtype=np.int8)
        for s in range(block.shape[1]):
            h0, h1 = phase_local_ancestry_sample(block[:, s, 0], block[:, s, 1], refs, config)
            out[:, s, 0] = h0
            out[:, s, 1] = h1
        return out

    per_sample = hap.rechunk((L, 1, 2))
    return da.map_blocks(_phase_block, per_sample, dtype=np.int8, chunks=per_sample.chunks)


def phase_dataset(
    ds: "xr.Dataset", ref_zarr_root: Optional[str] = None,
    sample_annot_path: Optional[str] = None, *, method: str = "gnomix",
    config: Optional[PhasingConfig] = None, groups: Optional[list[str]] = None,
    hap_index_in_zarr: int = 0,
) -> "xr.Dataset":
    """
    Phase-correct a local-ancestry Dataset (one chromosome).

    Returns a new Dataset whose ``haplotype_ancestry`` (and ``posterior``, when
    present) carry the corrected orientation, plus a boolean
    ``phase_swapped (variant, sample)`` variable marking the exchanged loci.

    Parameters
    ----------
    method : {"gnomix", "reference"}
        See :func:`phase_haplotypes`.  ``"gnomix"`` uses ``ds.la.posterior``
        when the Dataset has posteriors (``open_rfmix(source="fb",
        keep_posteriors=True)``) and one-hot codes otherwise.
    """
    from ..core import schema as S

    chroms = list(dict.fromkeys(str(c) for c in ds[S.CHROMOSOME].values))
    if len(chroms) != 1:
        raise ValueError(
            f"phase_dataset expects a single chromosome; got {chroms}. "
            "Use ds.la.sel_region(chrom) first."
        )
    config = config or PhasingConfig()
    hap = ds[S.HAPLOTYPE_ANCESTRY].data
    post = ds[S.POSTERIOR].data if S.POSTERIOR in ds else None

    if method == "gnomix":
        import dask.array as da
        swapped = gnomix_switch_mask(hap, post, config=config)
        swapped = swapped.rechunk((da.asarray(hap).chunks[0], da.asarray(hap).chunks[1]))
        codes, post_c = _apply_swaps(hap, swapped, post)
    elif method == "reference":
        positions = np.asarray(ds[S.VARIANT_POSITION].values, dtype=np.int64)
        codes = phase_haplotypes(
            hap, positions, chroms[0], ref_zarr_root, sample_annot_path, config=config,
            groups=groups, hap_index_in_zarr=hap_index_in_zarr, method="reference",
        )
        swapped = (codes[:, :, 0] != hap[:, :, 0]) & (hap[:, :, 0] >= 0)
        post_c = _apply_swaps(hap, swapped, post)[1] if post is not None else None
    else:
        raise ValueError("method must be 'gnomix' or 'reference'.")

    new_vars = {S.HAPLOTYPE_ANCESTRY: ((S.VARIANT, S.SAMPLE, S.PLOIDY), codes),
                "phase_swapped": ((S.VARIANT, S.SAMPLE), swapped)}
    if post_c is not None:
        new_vars[S.POSTERIOR] = ((S.VARIANT, S.SAMPLE, S.PLOIDY, S.ANCESTRY), post_c)
    out = ds.assign(new_vars)
    out.attrs = dict(ds.attrs)
    out.attrs["phased"] = True
    out.attrs["phasing_method"] = method
    return out


def phase_rfmix_chromosome_to_zarr(
    file_prefix: str, ref_zarr_root: Optional[str], sample_annot_path: Optional[str],
    output_path: str, *, chrom: Optional[str] = None,
    groups: Optional[list[str]] = None,
    config: Optional[PhasingConfig] = None,
    hap_index_in_zarr: int = 0,
    method: str = "gnomix",
    source: str = "fb",
    keep_posteriors: bool = True,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
    binary_dir: Optional[str] = None,
    generate_binary: Optional[bool] = None,
) -> xr.Dataset:
    """
    Phase local ancestry for a single chromosome and write a Zarr store.

    The store follows :mod:`rfmix_reader.core.schema` (``haplotype_ancestry``
    holds the corrected codes) and can be reopened with
    :func:`rfmix_reader.open_local_ancestry`.

    Parameters
    ----------
    file_prefix
        RFMix outputs (directory, file or prefix).
    ref_zarr_root, sample_annot_path
        Reference VCF-Zarr store(s) and the ``sample_id``/``group`` table;
        only used (and required) by ``method="reference"``.
    output_path
        Destination Zarr store.
    method
        ``"gnomix"`` (default, posterior-based) or ``"reference"``.
    keep_posteriors
        ``source="fb"`` only: read the posteriors so the gnomix method can
        use them (recommended).
    chrom
        Chromosome to phase; required when ``file_prefix`` holds several.
    source
        ``"fb"`` (default) or ``"msp"``.
    cache_dir
        Optional cache for the unphased Dataset (see :func:`open_rfmix`).
    binary_dir, generate_binary
        Ignored; kept for backwards compatibility with the ``.bin`` workflow.
    """
    from ..core.api import open_rfmix

    if binary_dir is not None or generate_binary is not None:
        logger.warning("binary_dir/generate_binary are ignored; the Dataset reader streams "
                       "the source directly (use cache_dir to persist it).")
    config = config or PhasingConfig()
    logger.info("[phase_rfmix_chromosome_to_zarr] Starting phasing for chrom=%s into %s",
                chrom if chrom is not None else "all", output_path)

    ds = open_rfmix(file_prefix, source=source, chrom=chrom, cache_dir=cache_dir,
                    keep_posteriors=(keep_posteriors and source == "fb"), verbose=verbose)
    chroms = ds.la.chromosomes
    if len(chroms) != 1:
        raise ValueError("Phasing per chromosome expects a single chromosome in the input.")
    logger.info("[phase_rfmix_chromosome_to_zarr] Loaded %d variants across %d samples",
                ds.la.n_variants, ds.la.n_samples)

    phased = phase_dataset(ds, ref_zarr_root, sample_annot_path, method=method, config=config,
                           groups=groups, hap_index_in_zarr=hap_index_in_zarr)
    logger.info("[phase_rfmix_chromosome_to_zarr] Writing phased data to %s", output_path)
    _write_schema_zarr(phased, output_path)
    return phased


def _write_schema_zarr(ds: xr.Dataset, output_path: str) -> None:
    """Write a schema Dataset with ``to_zarr`` (string coords as variable-length UTF-8)."""
    out = ds.copy()
    for name in list(out.coords) + list(out.data_vars):
        var = out[name]
        var.encoding.pop("chunks", None)
        var.encoding.pop("preferred_chunks", None)
        if var.dtype.kind in ("U", "O"):
            values = to_str_array(var.values)
            out = out.assign_coords({name: (var.dims, values)}) if name in out.coords \
                else out.assign({name: (var.dims, values)})
    out.to_zarr(output_path, mode="w", consolidated=False)


def merge_phased_zarrs(
    chrom_zarr_paths: List[str], output_path: str, *, sort: bool = True
) -> xr.Dataset:
    """
    Merge per-chromosome phased Zarr stores (schema Datasets) along ``variant``.

    Parameters
    ----------
    chrom_zarr_paths
        Stores written by :func:`phase_rfmix_chromosome_to_zarr`.
    output_path
        Destination Zarr store for the merged Dataset.
    sort
        Sort the stores by chromosome before concatenating.
    """
    from ..core import schema as S
    from ..core.zarr_io import open_store
    from ..formats.common import chrom_sort_key

    if not chrom_zarr_paths:
        raise ValueError("No Zarr paths provided for merging.")
    logger.info("[merge_phased_zarrs] Opening %d per-chromosome Zarr stores", len(chrom_zarr_paths))
    paths = [Path(p) for p in chrom_zarr_paths]
    if sort:
        paths = sorted(paths, key=lambda p: chrom_sort_key(p.stem))
    combined = S.concat_datasets([open_store(p) for p in paths])
    logger.info("[merge_phased_zarrs] Writing merged dataset with %d variants to %s",
                combined.sizes.get(S.VARIANT, 0), output_path)
    _write_schema_zarr(combined, output_path)
    return combined
