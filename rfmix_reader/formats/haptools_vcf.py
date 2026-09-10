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

from ..readers.read_simu import _get_ancestry_labels, _init_vcf, _map_pop_to_codes
from .base import MISSING, Chunk, Header

FORMAT = "haptools"


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
        codes = np.where(codes == MISSING.astype(np.int16) & 0xFF, -1, codes)  # 255 -> -1
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
