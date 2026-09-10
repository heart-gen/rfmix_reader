"""Reference-panel conversion helpers (``prepare-reference``)."""
from __future__ import annotations

__all__ = ["convert_vcf_to_zarr", "convert_vcfs_to_zarr"]

_lazy = {
    "convert_vcf_to_zarr": (".prepare_reference", "convert_vcf_to_zarr"),
    "convert_vcfs_to_zarr": (".prepare_reference", "convert_vcfs_to_zarr"),
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
