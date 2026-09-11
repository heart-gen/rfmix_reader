"""
The local-ancestry Dataset core: schema, ``ds.la`` accessor, Zarr cache and the
``open_*`` readers.  Importing this package registers the accessor.

The reader and Zarr-store names are resolved lazily: they import the
``formats`` package, whose leaf module ``formats.base`` imports
:mod:`rfmix_reader.core.codes`, so either package can be imported first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import accessor  # noqa: F401  (registers ``ds.la``)
from .accessor import LocalAncestryAccessor
from .schema import MISSING, build_dataset, concat_datasets, validate

__all__ = [
    "LocalAncestryAccessor",
    "MISSING",
    "build_dataset", "concat_datasets", "validate",
    "open_store", "store_path", "write_store",
    "open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert",
]

_lazy = {
    "open_rfmix": (".api", "open_rfmix"),
    "open_flare": (".api", "open_flare"),
    "open_simu": (".api", "open_simu"),
    "open_local_ancestry": (".api", "open_local_ancestry"),
    "convert": (".api", "convert"),
    "open_store": (".zarr_io", "open_store"),
    "store_path": (".zarr_io", "store_path"),
    "write_store": (".zarr_io", "write_store"),
}


def __getattr__(name: str):
    if name in _lazy:
        import importlib

        mod_name, attr_name = _lazy[name]
        obj = getattr(importlib.import_module(mod_name, __name__), attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .api import convert, open_flare, open_local_ancestry, open_rfmix, open_simu
    from .zarr_io import open_store, store_path, write_store
