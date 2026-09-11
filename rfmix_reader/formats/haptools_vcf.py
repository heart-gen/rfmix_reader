"""
haptools ``simgenotype`` VCF parser (``POP`` FORMAT field with two labels).

Regions are pulled through the tabix index in parallel threads and consumed in
genomic order.  haptools defines no population order, so ancestries are the
sorted labels from the ``.bp`` file (or the first records of the VCF).
"""
from __future__ import annotations

import multiprocessing as mp
from collections import deque
from concurrent.futures import BrokenExecutor, Executor, ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
from typing import Dict, Iterator, Optional

import numpy as np
from cyvcf2 import VCF

from os.path import basename, dirname, exists, join
from pathlib import Path
from re import sub
from typing import List, Tuple

from .base import Chunk, Header
from .common import filter_paths_by_chrom

#: haplotype code for a population label that is not in the ancestry list
POP_MISSING = np.uint8(255)

FORMAT = "haptools"


def _normalize_labels(arr: np.ndarray | List[str]) -> np.ndarray:
    """
    Normalize ancestry labels for consistent mapping.
    """
    arr = np.array(arr, dtype="U")
    arr = np.char.strip(arr)
    arr = np.char.upper(arr)
    arr = np.char.replace(arr, " ", "")
    return arr


def _map_pop_to_codes(pop_mat: np.ndarray, ancestries: np.ndarray) -> np.ndarray:
    """
    Map ancestry labels in pop_mat (strings) to numeric codes.
    Uses binary search on sorted ancestry list.
    """
    # Flatten haplotypes
    parts = np.char.partition(pop_mat, ",")
    h0, h1 = parts[:, :, 0], parts[:, :, 2]

    # Normalize both haplotype arrays
    hap = np.stack([h0, h1], axis=-1)
    hap = _normalize_labels(hap)

    # Fast searchsorted lookup
    idx = np.searchsorted(ancestries, hap)
    idx = np.clip(idx, 0, len(ancestries) - 1)
    valid = ancestries[idx] == hap
    codes = np.where(valid, idx.astype(np.uint8), POP_MISSING)

    return codes


def _parse_pop_labels(vcf_file: str, max_records: int = 100) -> List[str]:
    """
    Parse ancestry population labels from a breakpoint (.bp) file
    or from the VCF POP FORMAT field if .bp is missing.

    Ensure VCF is BGZF-compressed and tabix-indexed.
    """
    # Derive .bp file path from VCF path
    vcf_dir = dirname(vcf_file)
    base_name = basename(vcf_file)
    chr_prefix = sub(r"\.vcf\.gz$", "", base_name)
    bp_file = join(vcf_dir, f"{chr_prefix}.bp")

    ancestries = set()

    if exists(bp_file):
        # Primary: read from .bp file (faster)
        with open(bp_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Sample_"):
                    continue
                parts = line.split()
                if parts:
                    ancestries.add(parts[0])
    else:
        # Fallback
        vcf = VCF(vcf_file)
        try:
            n_scanned = 0
            for rec in vcf:
                try:
                    pop = rec.format("POP")
                except Exception:
                    continue

                if pop is not None:
                    flat = np.asarray(pop).astype(str).ravel()
                    for entry in flat:
                        if not entry:
                            continue
                        ancestries.update(entry.replace(" ", "").split(","))

                n_scanned += 1
                if n_scanned >= max_records:
                    break
        finally:
            vcf.close()

    if not ancestries:
        raise ValueError(
            f"No ancestry labels found in .bp file or first {max_records} "
            f"records of VCF: {vcf_file}"
        )

    # Normalize
    ancestries = _normalize_labels(list(ancestries))
    return sorted(set(ancestries))


def _build_mapper(ancestries: List[str]) -> Tuple[np.ndarray, dict[str, np.uint8]]:
    """
    Build fast ancestry lookup: returns sorted ancestry array + dict for labels.
    """
    ancestries = np.array(ancestries, dtype="U")
    mapper = {a: np.uint8(i) for i, a in enumerate(ancestries)}
    return ancestries, mapper


def _get_ancestry_labels(vcf_file):
    ancestries = _parse_pop_labels(vcf_file)
    return _build_mapper(ancestries)


def _get_vcf_files(vcf_path: str, chrom: Optional[str] = None) -> List[str]:
    """
    Resolve a path into a list of ancestry-annotated VCF files.

    Parameters
    ----------
    vcf_path : str
        Path to a directory containing `.vcf` or `.vcf.gz` files.
    chrom : str, optional
        Chromosome label used to filter the results.

    Returns
    -------
    list of str
        Sorted list of VCF file paths.

    Raises
    ------
    ValueError
        If `vcf_path` is not a valid file or directory.
    FileNotFoundError
        If no VCF files matching the pattern are found.
    """
    vcf_path = Path(vcf_path)

    if vcf_path.is_dir():
        candidates = sorted(vcf_path.glob("*.vcf*"))
    elif vcf_path.is_file() and vcf_path.suffix in {".vcf", ".gz"}:
        candidates = [vcf_path]
    else:
        raise ValueError(
            f"Invalid input: {vcf_path} must be a .vcf, .vcf.gz file, "
            f"or directory containing them."
        )

    # Filter out unwanted files
    vcf_files = []
    for f in candidates:
        suffixes = "".join(f.suffixes)
        if suffixes not in {".vcf", ".vcf.gz"}:
            continue  # skip things like .tbi
        if f.name.endswith("anc.vcf") or f.name.endswith("anc.vcf.gz"):
            continue
        vcf_files.append(f)

    if not vcf_files:
        raise FileNotFoundError(f"No VCF files found in path: {vcf_path}")

    return sorted(filter_paths_by_chrom([str(f) for f in vcf_files], chrom))


def _init_vcf(vcf_file, vcf_threads):
    vcf = VCF(vcf_file)
    try:
        if vcf_threads and hasattr(vcf, "set_threads"):
            vcf.set_threads(vcf_threads)
        samples = vcf.samples
        first = next(iter(vcf), None)
        if first is None:
            raise ValueError(f"VCF has no records: {vcf_file}")
        seqname = first.CHROM
        seqnames = list(vcf.seqnames)
        if seqname not in seqnames:
            raise ValueError(
                f"Contig '{seqname}' of the first record is not declared in the "
                f"header of {vcf_file}; add '##contig=<ID={seqname},length=...>' "
                "(see README: reheader before calling read_simu)."
            )
        seqlen = vcf.seqlens[seqnames.index(seqname)]
    finally:
        vcf.close()

    return None, samples, seqname, seqlen


def discover(path: str, chrom: Optional[str] = None):
    from .discover import discover as _discover

    return _discover(path, FORMAT, chrom)


def scan(filemap: Dict[str, str], n_threads: int = 4, **_) -> Header:
    fn = filemap["vcf"]
    _, samples, chrom, chrom_len = _init_vcf(fn, n_threads)
    ancestries, _ = _get_ancestry_labels(fn)
    return Header(
        samples=list(samples), ancestries=[str(a) for a in ancestries],
        source_files=[fn], chrom=str(chrom), n_variants=None, global_ancestry=None,
        extra={"chrom_len": int(chrom_len), "record_chrom": str(chrom)},
    )


def _pair_table(ancestries) -> Dict[str, Tuple[int, int]]:
    """Canonical ``"A,B"`` POP strings -> ``(code_a, code_b)`` for every ordered pair."""
    labels = [str(a) for a in ancestries]
    return {f"{a},{b}": (i, j) for i, a in enumerate(labels) for j, b in enumerate(labels)}


def _codes_from_pop_matrix(pop_mat: np.ndarray, ancestries: np.ndarray,
                           pairs: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """
    ``(n, S)`` unicode ``"A,B"`` matrix -> ``(n, S, 2)`` int8 codes (``-1`` unknown).

    One vectorised equality test per ordered ancestry pair (A^2 tests, each a
    memcmp over the matrix) replaces the per-element partition / strip / upper
    pipeline; only elements that match no canonical pair (odd spacing or case)
    go through the slow normalising path.
    """
    codes = np.full(pop_mat.shape + (2,), -1, dtype=np.int8)
    matched = np.zeros(pop_mat.shape, dtype=bool)
    for label, pair in pairs.items():
        m = pop_mat == label
        if m.any():
            codes[m] = pair
            matched |= m
    if not matched.all():
        rest = ~matched
        slow = _map_pop_to_codes(pop_mat[rest][:, None], ancestries)[:, 0].astype(np.int16)
        codes[rest] = np.where(slow == POP_MISSING, -1, slow).astype(np.int8)
    return codes


def _pull_region(fn: str, chrom: str, start: int, end: int, ancestries: Tuple[str, ...],
                 n_samples: int, vcf_threads: int):
    """Codes for one tabix region: ``(positions int32, codes int8 (n, S, 2))`` or ``None``."""
    anc = np.asarray(ancestries, dtype="U")
    pairs = _pair_table(anc)
    vcf = VCF(fn)
    try:
        if vcf_threads and hasattr(vcf, "set_threads"):
            vcf.set_threads(vcf_threads)
        positions, pops = [], []
        for r in vcf(f"{chrom}:{start}-{end}"):
            p = r.format("POP")
            if p is None:
                p = np.full(n_samples, "", dtype="U1")
            positions.append(r.POS)
            pops.append(p)
    finally:
        vcf.close()
    if not positions:
        return None
    pop_mat = np.stack(pops)                       # (n, S) unicode, decoded once by cyvcf2
    codes = _codes_from_pop_matrix(pop_mat, anc, pairs)
    return np.asarray(positions, dtype=np.int32), codes


def _region_pool(workers: int) -> Executor:
    """
    Executor for the region pulls.

    Worker *processes* only with the ``fork`` start method, which does not
    re-import the caller's ``__main__`` (so an unguarded script that calls
    ``open_simu`` at top level cannot spawn itself recursively), and never
    from a daemonic process (which may not have children) or with a single
    worker.  Everything else runs the pulls in threads.
    """
    if workers <= 1 or mp.current_process().daemon or "fork" not in mp.get_all_start_methods():
        return ThreadPoolExecutor(max_workers=workers)
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"))


def iter_chunks(filemap: Dict[str, str], header: Header, chunk_rows: int = 10_000,
                region_bp: int = 1_000_000, n_threads: int = 4, **_) -> Iterator[Chunk]:
    fn = filemap["vcf"]
    chrom = header.extra["record_chrom"]
    chrom_len = header.extra["chrom_len"]
    ancestries = tuple(str(a) for a in header.ancestries)
    n_samples = len(header.samples)

    def submit(pool, start: int):
        end = min(start + region_bp - 1, chrom_len)
        return pool.submit(_pull_region, fn, chrom, start, end, ancestries, n_samples, 1)

    starts = iter(range(1, chrom_len + 1, region_bp))
    pending_pos, pending_codes = [], []
    n_pending = 0

    def drain(force: bool) -> Iterator[Chunk]:
        nonlocal pending_pos, pending_codes, n_pending
        while n_pending and (force or n_pending >= chunk_rows):
            pos = np.concatenate(pending_pos)
            codes = np.concatenate(pending_codes)
            take = min(chunk_rows, len(pos))
            yield Chunk(np.array([chrom] * take, dtype=str), pos[:take], pos[:take].copy(), codes[:take])
            rest_pos, rest_codes = pos[take:], codes[take:]
            pending_pos = [rest_pos] if len(rest_pos) else []
            pending_codes = [rest_codes] if len(rest_codes) else []
            n_pending = len(rest_pos)

    # cyvcf2 decodes the POP FORMAT field per record under the GIL, so regions
    # are pulled in worker processes where that is safe (see _region_pool);
    # at most 2 x n_threads regions are in flight, consumed in genomic order.
    workers = max(1, n_threads)
    pool = _region_pool(workers)
    try:
        futures = deque(submit(pool, s) for s in islice(starts, 2 * workers))
        if futures:
            futures[0].result()                               # surfaces a broken pool early
    except (BrokenExecutor, OSError, RuntimeError, AssertionError, ValueError):
        pool.shutdown(wait=False, cancel_futures=True)
        pool = ThreadPoolExecutor(max_workers=workers)
        starts = iter(range(1, chrom_len + 1, region_bp))
        futures = deque(submit(pool, s) for s in islice(starts, 2 * workers))
    with pool:
        while futures:
            fut = futures.popleft()
            nxt = next(starts, None)
            if nxt is not None:
                futures.append(submit(pool, nxt))
            result = fut.result()
            if result is None:
                continue
            positions, codes = result
            pending_pos.append(positions)
            pending_codes.append(codes)
            n_pending += len(positions)
            yield from drain(force=False)
    yield from drain(force=True)
