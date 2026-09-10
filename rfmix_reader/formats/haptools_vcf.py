"""
haptools ``simgenotype`` VCF parser (``POP`` FORMAT field with two labels).

Regions are pulled through the tabix index in parallel threads and consumed in
genomic order.  haptools defines no population order, so ancestries are the
sorted labels from the ``.bp`` file (or the first records of the VCF).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def iter_chunks(filemap: Dict[str, str], header: Header, chunk_rows: int = 10_000,
                region_bp: int = 1_000_000, n_threads: int = 4, **_) -> Iterator[Chunk]:
    fn = filemap["vcf"]
    chrom = header.extra["record_chrom"]
    chrom_len = header.extra["chrom_len"]
    ancestries = np.asarray(header.ancestries, dtype="U")

    def pull(start: int):
        vcf = VCF(fn)
        try:
            if n_threads and hasattr(vcf, "set_threads"):
                vcf.set_threads(n_threads)
            end = min(start + region_bp - 1, chrom_len)
            recs = list(vcf(f"{chrom}:{start}-{end}"))
            if not recs:
                return None
            positions = np.fromiter((r.POS for r in recs), dtype=np.int32, count=len(recs))
            pop_mat = np.array([r.format("POP") for r in recs], dtype="U")
        finally:
            vcf.close()
        codes = _map_pop_to_codes(pop_mat, ancestries).astype(np.int16)
        codes = np.where(codes == int(POP_MISSING), -1, codes)  # 255 -> -1
        return positions, codes.astype(np.int8)

    starts = range(1, chrom_len + 1, region_bp)
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

    with ThreadPoolExecutor(max_workers=max(1, n_threads)) as pool:
        futures = [pool.submit(pull, s) for s in starts]
        for fut in futures:                     # genomic order
            result = fut.result()
            if result is None:
                continue
            positions, codes = result
            pending_pos.append(positions)
            pending_codes.append(codes)
            n_pending += len(positions)
            yield from drain(force=False)
    yield from drain(force=True)
