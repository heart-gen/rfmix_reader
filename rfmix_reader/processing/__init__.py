"""Processing utilities for rfmix_reader.

Heavy modules (``imputation`` needs zarr, ``phase`` needs xarray) are loaded
lazily on first attribute access.
"""

from __future__ import annotations

from .constants import CHROM_SIZES, COORDINATES

__all__ = [
    "CHROM_SIZES",
    "COORDINATES",
    "PhasingConfig",
    "interpolate_array",
    "phase_admix_dask_with_index",
    "phase_rfmix_chromosome_to_zarr",
    "merge_phased_zarrs",
]

_lazy = {
    "interpolate_array": (".imputation", "interpolate_array"),
    "PhasingConfig": (".phase", "PhasingConfig"),
    "phase_admix_dask_with_index": (".phase", "phase_admix_dask_with_index"),
    "phase_rfmix_chromosome_to_zarr": (".phase", "phase_rfmix_chromosome_to_zarr"),
    "merge_phased_zarrs": (".phase", "merge_phased_zarrs"),
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
