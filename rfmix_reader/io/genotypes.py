"""
Genotype calls and likelihoods from a VCF/BCF, in the layout LoGicAl expects.

LoGicAl (Wang & Zöllner, 2026) estimates ancestry-specific allele frequencies
one site at a time from, per individual, the two haplotype ancestry labels
(or probabilities), the best-guess genotype ``g`` (ALT dosage 0/1/2), the
phased alleles ``h1``/``h2`` (0/1) and a 4-column genotype-likelihood row
ordered by the phased allele pair ``(h1, h2) = (0,0), (1,0), (0,1), (1,1)``.
This module reads a VCF with cyvcf2 and produces those arrays, block by block,
for :func:`rfmix_reader.ops.logical.to_logical`.

Missing values are ``-1`` in the int8 arrays and ``NaN`` in the likelihoods.
Multi-allelic records are skipped (split them with ``bcftools norm -m -any``
first); records without an ALT allele are skipped too.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import chain
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..formats.common import normalize_chrom_label

__all__ = ["GenotypeBlock", "iter_genotype_blocks", "gt_to_calls", "pl_to_gl4", "gl_to_gl4",
           "vcf_samples"]

logger = logging.getLogger(__name__)

#: Column order of the 4-column likelihood row: (h1, h2) allele pairs.
GL4_COLUMNS = ("gl_00", "gl_10", "gl_01", "gl_11")


@dataclass
class GenotypeBlock:
    """One block of biallelic records from the VCF (see :func:`iter_genotype_blocks`)."""

    chrom: np.ndarray            #: (n,) str, the VCF contig name
    pos: np.ndarray              #: (n,) int32, 1-based position
    vid: np.ndarray              #: (n,) object str, the ID column ("." when absent)
    ref: np.ndarray              #: (n,) object str
    alt: np.ndarray              #: (n,) object str
    is_snp: np.ndarray           #: (n,) bool
    g: np.ndarray                #: (n, S) int8 ALT dosage 0/1/2, -1 missing
    h1: np.ndarray               #: (n, S) int8 ALT allele on haplotype 1, -1 unknown
    h2: np.ndarray               #: (n, S) int8 ALT allele on haplotype 2, -1 unknown
    gl: Optional[np.ndarray]     #: (n, S, 4) float32 linear-scale likelihoods, NaN missing; None if no PL/GL
    samples: List[str]           #: sample names in the order of the S axis (the VCF's order)
    skipped: List[Tuple[int, str, str, str]] = field(default_factory=list)  #: (pos, ref, alt, reason)

    def __len__(self) -> int:
        return int(self.pos.size)


def gt_to_calls(gt_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ``(S, 3)`` cyvcf2 genotype array ``[allele1, allele2, phased]`` ->
    ``(g, h1, h2)`` int8 vectors.

    ``g`` is the ALT dosage (``-1`` when either allele is missing).  ``h1``/``h2``
    are the alleles of haplotype 1 and 2 when the call is phased or homozygous
    (phase is irrelevant for a homozygote), and ``-1`` for an unphased
    heterozygote.  Alleles above 1 (multi-allelic records) are treated as
    missing; callers skip such records before reaching here.
    """
    arr = np.asarray(gt_array)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"gt_array must be (S, >=2); got {arr.shape}.")
    a1 = arr[:, 0].astype(np.int16)
    a2 = arr[:, 1].astype(np.int16)
    phased = arr[:, 2].astype(bool) if arr.shape[1] > 2 else np.zeros(arr.shape[0], dtype=bool)
    ok = (a1 >= 0) & (a2 >= 0) & (a1 <= 1) & (a2 <= 1)
    g = np.where(ok, a1 + a2, -1).astype(np.int8)
    known = ok & (phased | (a1 == a2))
    h1 = np.where(known, a1, -1).astype(np.int8)
    h2 = np.where(known, a2, -1).astype(np.int8)
    return g, h1, h2


def pl_to_gl4(pl: np.ndarray) -> np.ndarray:
    """
    ``(S, 3)`` Phred-scaled likelihoods (``PL``) -> ``(S, 4)`` float32 linear
    likelihoods in LoGicAl's phased-pair order ``(0,0), (1,0), (0,1), (1,1)``.

    The heterozygote likelihood is copied into both middle columns because an
    unphased ``PL`` cannot distinguish ``1|0`` from ``0|1``.  Any negative
    value (cyvcf2's missing sentinel) marks the whole row as missing (NaN).
    Values are not normalised: LoGicAl's EM divides by the row sum.
    """
    p = np.asarray(pl)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pl must be (S, 3); got {p.shape}.")
    pf = p.astype(np.float64)
    missing = (pf < 0).any(axis=1) | ~np.isfinite(pf).all(axis=1)
    lin = np.power(10.0, -np.where(missing[:, None], 0.0, pf) / 10.0)
    out = np.stack([lin[:, 0], lin[:, 1], lin[:, 1], lin[:, 2]], axis=1).astype(np.float32)
    out[missing] = np.nan
    return out


def gl_to_gl4(gl: np.ndarray) -> np.ndarray:
    """
    ``(S, 3)`` log10 genotype likelihoods (``GL``) -> ``(S, 4)`` float32 linear
    likelihoods in LoGicAl's phased-pair order (see :func:`pl_to_gl4`).
    Non-finite rows are missing (NaN).
    """
    g = np.asarray(gl, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gl must be (S, 3); got {g.shape}.")
    missing = ~np.isfinite(g).all(axis=1)
    lin = np.power(10.0, np.where(missing[:, None], 0.0, g))
    out = np.stack([lin[:, 0], lin[:, 1], lin[:, 1], lin[:, 2]], axis=1).astype(np.float32)
    out[missing] = np.nan
    return out


def vcf_samples(vcf_path) -> List[str]:
    """Sample names of a VCF/BCF, in file order."""
    from cyvcf2 import VCF

    vcf = VCF(str(vcf_path))
    try:
        return [str(s) for s in vcf.samples]
    finally:
        vcf.close()


def _matching_contigs(vcf, chrom: str) -> List[str]:
    """Contig names declared in the VCF that equal ``chrom`` up to a ``chr`` prefix."""
    target = normalize_chrom_label(str(chrom))
    return [c for c in vcf.seqnames if normalize_chrom_label(str(c)) == target]


def _record_iterator(vcf, chrom: Optional[str], region: Optional[Tuple[int, int]]):
    """Records on ``chrom`` (any contig spelling) within ``region``, through the index when possible."""
    if chrom is None:
        if region is not None:
            raise ValueError("region requires chrom.")
        return iter(vcf)
    contigs = _matching_contigs(vcf, chrom)
    target = normalize_chrom_label(str(chrom))

    def filtered():
        lo, hi = region if region is not None else (None, None)
        for rec in vcf:
            if normalize_chrom_label(str(rec.CHROM)) != target:
                continue
            if lo is not None and (rec.POS < lo or rec.POS > hi):
                continue
            yield rec

    if not contigs:
        return filtered()           # contig not declared in the header: scan
    query = contigs[0] if region is None else f"{contigs[0]}:{int(region[0])}-{int(region[1])}"
    # cyvcf2's region query is a generator: the "no index" failure surfaces on
    # the first next(), so probe it here and fall back to a scan.
    try:
        it = iter(vcf(query))
        first = next(it)
    except StopIteration:
        return iter(())
    except Exception:               # no index (plain .vcf, or .bcf without .csi): scan
        return filtered()
    return chain([first], it)


def iter_genotype_blocks(
    vcf_path, *, chrom: Optional[str] = None, samples: Optional[Sequence[str]] = None,
    region: Optional[Tuple[int, int]] = None, block_size: int = 20_000,
    snps_only: bool = False, gl_field: Optional[str] = "auto",
) -> Iterator[GenotypeBlock]:
    """
    Stream biallelic records of a VCF/BCF as :class:`GenotypeBlock` objects.

    Parameters
    ----------
    vcf_path : path
        VCF (``.vcf``, ``.vcf.gz``) or BCF.  A tabix/CSI index lets ``chrom``
        and ``region`` use random access; without one the file is scanned.
    chrom : str, optional
        Keep records of this chromosome only (``chr21`` and ``21`` both match).
    samples : sequence of str, optional
        Restrict to these samples.  cyvcf2 returns them in **file order**,
        which :attr:`GenotypeBlock.samples` reports.
    region : (start, end), optional
        1-based inclusive position range on ``chrom``.
    block_size : int
        Records per block (before skipping).
    snps_only : bool
        Skip records that are not single-nucleotide substitutions.
    gl_field : {"auto", "PL", "GL", None}
        FORMAT field to turn into the 4-column likelihoods.  ``"auto"`` picks
        ``PL`` if the header declares it, else ``GL``, else none.

    Yields
    ------
    GenotypeBlock
        Multi-allelic records, records without an ALT allele and (with
        ``snps_only``) non-SNPs are listed in :attr:`GenotypeBlock.skipped`
        with a reason (``multiallelic``, ``no_alt``, ``non_snp``).
    """
    from cyvcf2 import VCF

    kwargs = {"samples": [str(s) for s in samples]} if samples is not None else {}
    vcf = VCF(str(vcf_path), **kwargs)
    try:
        names = [str(s) for s in vcf.samples]
        if samples is not None and not names:
            raise ValueError("None of the requested samples is in the VCF.")
        if gl_field == "auto":
            gl_field = "PL" if vcf.contains("PL") else ("GL" if vcf.contains("GL") else None)
        elif gl_field is not None and not vcf.contains(gl_field):
            raise ValueError(f"FORMAT field {gl_field!r} is not declared in the VCF header.")
        convert = {"PL": pl_to_gl4, "GL": gl_to_gl4, None: None}[gl_field]
        S = len(names)

        def flush(buf) -> GenotypeBlock:
            n = len(buf["pos"])
            gl = None
            if convert is not None:
                gl = (np.stack(buf["gl"]) if n else np.zeros((0, S, 4), dtype=np.float32))
            return GenotypeBlock(
                chrom=np.asarray(buf["chrom"], dtype=object), pos=np.asarray(buf["pos"], dtype=np.int32),
                vid=np.asarray(buf["vid"], dtype=object), ref=np.asarray(buf["ref"], dtype=object),
                alt=np.asarray(buf["alt"], dtype=object), is_snp=np.asarray(buf["is_snp"], dtype=bool),
                g=(np.stack(buf["g"]) if n else np.zeros((0, S), dtype=np.int8)),
                h1=(np.stack(buf["h1"]) if n else np.zeros((0, S), dtype=np.int8)),
                h2=(np.stack(buf["h2"]) if n else np.zeros((0, S), dtype=np.int8)),
                gl=gl, samples=list(names), skipped=list(buf["skipped"]),
            )

        def empty():
            return {k: [] for k in ("chrom", "pos", "vid", "ref", "alt", "is_snp", "g", "h1", "h2", "gl", "skipped")}

        buf = empty()
        seen = 0
        for rec in _record_iterator(vcf, chrom, region):
            alts = list(rec.ALT)
            alt_str = ",".join(alts) if alts else "."
            if len(alts) != 1:
                buf["skipped"].append((int(rec.POS), str(rec.REF), alt_str, "no_alt" if not alts else "multiallelic"))
            elif snps_only and not rec.is_snp:
                buf["skipped"].append((int(rec.POS), str(rec.REF), alt_str, "non_snp"))
            else:
                g, h1, h2 = gt_to_calls(rec.genotype.array())
                buf["chrom"].append(str(rec.CHROM))
                buf["pos"].append(int(rec.POS))
                buf["vid"].append(str(rec.ID) if rec.ID else ".")
                buf["ref"].append(str(rec.REF))
                buf["alt"].append(alts[0])
                buf["is_snp"].append(bool(rec.is_snp))
                buf["g"].append(g)
                buf["h1"].append(h1)
                buf["h2"].append(h2)
                if convert is not None:
                    try:
                        raw = rec.format(gl_field)
                    except KeyError:
                        raw = None
                    if raw is None or np.asarray(raw).shape != (S, 3):
                        buf["gl"].append(np.full((S, 4), np.nan, dtype=np.float32))
                    else:
                        buf["gl"].append(convert(np.asarray(raw)))
            seen += 1
            if seen >= block_size:
                yield flush(buf)
                buf, seen = empty(), 0
        if seen:
            yield flush(buf)
    finally:
        vcf.close()
