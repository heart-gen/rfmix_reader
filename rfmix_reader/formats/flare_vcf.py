"""FLARE ``.anc.vcf.gz`` parser (``AN1``/``AN2`` haplotype codes)."""
from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np
from cyvcf2 import VCF

from ..readers._common import align_g_anc_columns, pops_by_code
from ..readers.read_flare import _parse_ancestry_header
from .base import Chunk, Header, chrom_label_from_path, codes_from_pairs
from .global_ancestry import frame_to_array, read_flare_global

FORMAT = "flare"


def discover(path: str, chrom: Optional[str] = None):
    from .discover import discover as _discover

    return _discover(path, FORMAT, chrom)


def scan(filemap: Dict[str, str], **_) -> Header:
    fn = filemap["anc.vcf"]
    pops = pops_by_code(_parse_ancestry_header(fn))
    vcf = VCF(fn)
    try:
        samples = list(vcf.samples)
        first = next(iter(vcf), None)
        chrom = chrom_label_from_path(fn) or (first.CHROM if first is not None else None)
    finally:
        vcf.close()

    global_anc = None
    if "global.anc" in filemap:
        g = align_g_anc_columns(read_flare_global(filemap["global.anc"]), pops)
        global_anc = frame_to_array(g, samples, pops)

    return Header(
        samples=samples, ancestries=pops, source_files=sorted(filemap.values()),
        chrom=chrom, n_variants=None, global_ancestry=global_anc,
    )


def iter_chunks(filemap: Dict[str, str], header: Header, chunk_rows: int = 10_000, **_
                ) -> Iterator[Chunk]:
    fn = filemap["anc.vcf"]
    n_anc = header.n_ancestries

    def flush(chroms, positions, an1s, an2s) -> Chunk:
        codes = codes_from_pairs(np.stack(an1s), np.stack(an2s), n_anc)
        pos = np.asarray(positions, dtype=np.int32)
        return Chunk(np.asarray(chroms, dtype=str), pos, pos.copy(), codes)

    vcf = VCF(fn)
    try:
        chroms, positions, an1s, an2s = [], [], [], []
        for rec in vcf:
            chroms.append(rec.CHROM)
            positions.append(rec.POS)
            an1s.append(np.asarray(rec.format("AN1"), dtype=np.int32).ravel())
            an2s.append(np.asarray(rec.format("AN2"), dtype=np.int32).ravel())
            if len(chroms) == chunk_rows:
                yield flush(chroms, positions, an1s, an2s)
                chroms, positions, an1s, an2s = [], [], [], []
        if chroms:
            yield flush(chroms, positions, an1s, an2s)
    finally:
        vcf.close()
