"""I/O utilities and helpers for rfmix_reader.

Names are loaded lazily so that importing the light-weight pieces (``Chunk``,
``BinaryFileNotFoundError``) does not pull in zarr, pyarrow or xarray.
"""

from __future__ import annotations

from .chunk import Chunk
from .errors import BinaryFileNotFoundError

__all__ = [
    "Chunk",
    "BinaryFileNotFoundError",
    "admix_to_bed_individual",
    "write_data",
    "write_imputed",
]

_lazy = {
    "admix_to_bed_individual": (".loci_bed", "admix_to_bed_individual"),
    "write_data": (".write_data", "write_data"),
    "write_imputed": (".write_data", "write_imputed"),
}


def __getattr__(name: str):
    if name in _lazy:
        import importlib

        mod_name, attr_name = _lazy[name]
        mod = importlib.import_module(mod_name, __name__)
        obj = getattr(mod, attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
