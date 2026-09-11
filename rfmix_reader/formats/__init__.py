"""Streaming parsers for the supported local-ancestry formats."""
from __future__ import annotations

from .base import Chunk, Header
from .discover import FORMATS, discover, get_parser, primary_file
from ..core.codes import MISSING  # after the leaf modules above, so either package can be imported first

__all__ = ["Chunk", "Header", "MISSING", "FORMATS", "discover", "primary_file", "get_parser"]
