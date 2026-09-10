"""FLARE ``.anc.vcf.gz`` parser (``AN1``/``AN2`` haplotype codes)."""
from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np
from cyvcf2 import VCF

from re import search

import pandas as pd

from .common import align_g_anc_columns, pops_by_code
from .base import Chunk, Header, chrom_label_from_path, codes_from_pairs
from .global_ancestry import frame_to_array, read_flare_global

FORMAT = "flare"


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
