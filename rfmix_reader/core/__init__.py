"""
The local-ancestry Dataset core: schema, ``ds.la`` accessor, Zarr cache and the
``open_*`` readers.  Importing this package registers the accessor.
"""
from __future__ import annotations

from . import accessor  # noqa: F401  (registers ``ds.la``)
from .accessor import LocalAncestryAccessor
from .api import convert, open_flare, open_local_ancestry, open_rfmix, open_simu
from .schema import MISSING, build_dataset, concat_datasets, validate
from .zarr_io import open_store, store_path, write_store

__all__ = [
    "LocalAncestryAccessor",
    "MISSING",
    "build_dataset", "concat_datasets", "validate",
    "open_store", "store_path", "write_store",
    "open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert",
]
