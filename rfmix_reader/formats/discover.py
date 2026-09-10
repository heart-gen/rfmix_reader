"""Locate source files for a format and group them per chromosome."""
from __future__ import annotations

from typing import Dict, List, Optional

from ..utils import filter_file_maps_by_chrom, get_prefixes

__all__ = ["discover", "FORMATS"]

FORMATS = ("msp", "fb", "flare", "haptools")

_PRIMARY = {"msp": "msp.tsv", "fb": "fb.tsv", "flare": "anc.vcf", "haptools": "vcf"}


def primary_file(fmt: str, filemap: Dict[str, str]) -> str:
    return filemap[_PRIMARY[fmt]]


def discover(path: str, fmt: str, chrom: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Per-chromosome file maps for ``fmt`` under ``path``.

    Keys: ``msp.tsv``/``rfmix.Q`` (msp), ``fb.tsv``/``rfmix.Q`` (fb),
    ``anc.vcf``/``global.anc`` (flare), ``vcf`` (haptools).
    """
    if fmt == "msp":
        maps = get_prefixes(path, "msp", verbose=False)
        return filter_file_maps_by_chrom(maps, chrom, kind="MSP")
    if fmt == "fb":
        maps = get_prefixes(path, "rfmix", verbose=False)
        return filter_file_maps_by_chrom(maps, chrom, kind="RFMix")
    if fmt == "flare":
        maps = get_prefixes(path, "flare", verbose=False)
        return filter_file_maps_by_chrom(maps, chrom, kind="FLARE")
    if fmt == "haptools":
        from ..readers.read_simu import _get_vcf_files

        return [{"vcf": f} for f in _get_vcf_files(path, chrom=chrom)]
    raise ValueError(f"Unknown format {fmt!r}; choose from {FORMATS}.")
