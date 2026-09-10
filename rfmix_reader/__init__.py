"""
rfmix_reader: fast, lazy access to local-ancestry output (RFMix, FLARE,
haptools) as an xarray Dataset backed by a per-chromosome Zarr cache.
"""
from __future__ import annotations

from importlib.metadata import version as _v, PackageNotFoundError
from typing import TYPE_CHECKING

try:
    __version__ = _v("rfmix-reader")  # distribution name
except PackageNotFoundError:
    try:
        from ._version import __version__  # fallback for local builds
    except Exception:
        __version__ = "0.0.0"

# Public API
__all__ = [
    # readers / cache
    "open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert",
    # dataset helpers
    "build_dataset", "validate", "concat_datasets", "from_legacy", "MISSING",
    # processing
    "PhasingConfig", "phase_dataset", "merge_phased_zarrs", "interpolate_array",
    "CHROM_SIZES", "COORDINATES",
    # visualisation
    "save_multi_format", "plot_global_ancestry", "plot_ancestry_by_chromosome",
    "plot_local_ancestry_tagore",
]

# Map public names for lazy loading
_lazy = {
    "open_rfmix": (".core.api", "open_rfmix"),
    "open_flare": (".core.api", "open_flare"),
    "open_simu": (".core.api", "open_simu"),
    "open_local_ancestry": (".core.api", "open_local_ancestry"),
    "convert": (".core.api", "convert"),
    "build_dataset": (".core.schema", "build_dataset"),
    "validate": (".core.schema", "validate"),
    "concat_datasets": (".core.schema", "concat_datasets"),
    "from_legacy": (".core.legacy", "from_legacy"),
    "MISSING": (".core.codes", "MISSING"),
    "PhasingConfig": (".processing.phase", "PhasingConfig"),
    "phase_dataset": (".processing.phase", "phase_dataset"),
    "merge_phased_zarrs": (".processing.phase", "merge_phased_zarrs"),
    "interpolate_array": (".processing.imputation", "interpolate_array"),
    "CHROM_SIZES": (".processing.constants", "CHROM_SIZES"),
    "COORDINATES": (".processing.constants", "COORDINATES"),
    "save_multi_format": (".viz.visualization", "save_multi_format"),
    "plot_global_ancestry": (".viz.visualization", "plot_global_ancestry"),
    "plot_ancestry_by_chromosome": (".viz.visualization", "plot_ancestry_by_chromosome"),
    "plot_local_ancestry_tagore": (".viz.tagore", "plot_local_ancestry_tagore"),
}

_REMOVED = {
    "read_rfmix": "open_rfmix(...) and ds.la.to_legacy()",
    "read_rfmix_fb": 'open_rfmix(..., source="fb", cache_dir=...)',
    "read_flare": "open_flare(...)",
    "read_simu": "open_simu(...)",
    "read_fb": 'open_rfmix(..., source="fb")',
    "extract_locus_ancestry": "ds.la.at_positions(...)",
    "write_data": "ds.la.to_parquet(...)",
    "admix_to_bed_individual": "ds.la.to_bed(...)",
    "generate_tagore_bed": "ds.la.to_tagore(...)",
    "create_binaries": 'convert(path, "fb", cache_dir)',
    "Chunk": "chunk_rows= in open_* / convert",
    "BinaryFileNotFoundError": "the Zarr cache (no .bin files)",
    "get_pops": "ds.la.ancestries",
    "get_sample_names": "ds.la.samples",
    "get_prefixes": "rfmix_reader.formats.discover",
    "set_gpu_environment": "rfmix_reader.backends.describe_gpus",
    "delete_files_or_directories": "shutil",
    "phase_rfmix_chromosome_to_zarr": "rfmix_reader.processing.phase.phase_rfmix_chromosome_to_zarr",
}


def __getattr__(name: str):
    """Lazy attribute loader to keep import-time light."""
    if name in _lazy:
        import importlib
        mod_name, attr_name = _lazy[name]
        mod = importlib.import_module(mod_name, __name__)
        obj = getattr(mod, attr_name)
        globals()[name] = obj  # cache for future access
        return obj
    if name in _REMOVED:
        raise AttributeError(
            f"rfmix_reader.{name} was removed in 0.6; use {_REMOVED[name]} "
            "(see MIGRATION.md)."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # help() and tab-complete show public API
    return sorted(list(globals().keys()) + __all__)


# Make type checkers happy without importing heavy deps at runtime
if TYPE_CHECKING:
    from .core.api import convert, open_flare, open_local_ancestry, open_rfmix, open_simu
    from .core.codes import MISSING
    from .core.legacy import from_legacy
    from .core.schema import build_dataset, concat_datasets, validate
    from .processing.constants import CHROM_SIZES, COORDINATES
    from .processing.imputation import interpolate_array
    from .processing.phase import PhasingConfig, merge_phased_zarrs, phase_dataset
    from .viz.tagore import plot_local_ancestry_tagore
    from .viz.visualization import (
        plot_ancestry_by_chromosome,
        plot_global_ancestry,
        save_multi_format,
    )
