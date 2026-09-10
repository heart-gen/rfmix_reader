"""Streaming parsers for the supported local-ancestry formats."""
from __future__ import annotations

from importlib import import_module

from .base import Chunk, Header, MISSING
from .discover import FORMATS, discover, primary_file

__all__ = ["Chunk", "Header", "MISSING", "FORMATS", "discover", "primary_file", "get_parser"]

_MODULES = {
    "msp": ".rfmix_msp",
    "fb": ".rfmix_fb",
    "flare": ".flare_vcf",
    "haptools": ".haptools_vcf",
}


def get_parser(fmt: str):
    """Parser module for ``fmt`` (``msp``, ``fb``, ``flare`` or ``haptools``)."""
    if fmt not in _MODULES:
        raise ValueError(f"Unknown format {fmt!r}; choose from {FORMATS}.")
    return import_module(_MODULES[fmt], __name__)
