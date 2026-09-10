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
    "Chunk",
    "open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert",
    "read_fb", "read_simu", "read_rfmix", "read_rfmix_fb", "read_flare",
    "extract_locus_ancestry",
    "write_data",
    "admix_to_bed_individual",
    "CHROM_SIZES", "COORDINATES",
    "BinaryFileNotFoundError",
    "interpolate_array",
    "phase_rfmix_chromosome_to_zarr", "phase_dataset", "merge_phased_zarrs",
    "get_pops", "get_prefixes", "create_binaries", "get_sample_names",
    "set_gpu_environment", "delete_files_or_directories",
    "save_multi_format", "generate_tagore_bed",
    "plot_global_ancestry", "plot_ancestry_by_chromosome",
    "plot_local_ancestry_tagore",
]

# Map public names for lazy loading
_lazy = {
    "Chunk": (".io.chunk", "Chunk"),
    "open_rfmix": (".core.api", "open_rfmix"),
    "open_flare": (".core.api", "open_flare"),
    "open_simu": (".core.api", "open_simu"),
    "open_local_ancestry": (".core.api", "open_local_ancestry"),
    "convert": (".core.api", "convert"),
    "read_fb": (".readers.fb_read", "read_fb"),
    # legacy triple API (deprecated; implemented on the Dataset core)
    "read_simu": (".legacy", "read_simu"),
    "read_rfmix": (".legacy", "read_rfmix"),
    "extract_locus_ancestry": (".legacy", "extract_locus_ancestry"),
    "read_rfmix_fb": (".legacy", "read_rfmix_fb"),
    "read_flare": (".legacy", "read_flare"),
    "write_data": (".legacy", "write_data"),
    "admix_to_bed_individual": (".legacy", "admix_to_bed_individual"),
    "CHROM_SIZES": (".processing.constants", "CHROM_SIZES"),
    "COORDINATES": (".processing.constants", "COORDINATES"),
    "BinaryFileNotFoundError": (".io.errors", "BinaryFileNotFoundError"),
    "interpolate_array": (".processing.imputation", "interpolate_array"),
    "phase_rfmix_chromosome_to_zarr": (
        ".processing.phase",
        "phase_rfmix_chromosome_to_zarr",
    ),
    "phase_dataset": (".processing.phase", "phase_dataset"),
    "merge_phased_zarrs": (".processing.phase", "merge_phased_zarrs"),
    "get_pops": (".utils", "get_pops"),
    "get_prefixes": (".utils", "get_prefixes"),
    "create_binaries": (".legacy", "create_binaries"),
    "get_sample_names": (".utils", "get_sample_names"),
    "set_gpu_environment": (".utils", "set_gpu_environment"),
    "delete_files_or_directories": (".utils", "delete_files_or_directories"),
    "save_multi_format": (".viz.visualization", "save_multi_format"),
    "generate_tagore_bed": (".legacy", "generate_tagore_bed"),
    "plot_global_ancestry": (".viz.visualization", "plot_global_ancestry"),
    "plot_ancestry_by_chromosome": (".viz.visualization", "plot_ancestry_by_chromosome"),
    "plot_local_ancestry_tagore": (".viz.tagore", "plot_local_ancestry_tagore"),
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # help() and tab-complete show public API
    return sorted(list(globals().keys()) + __all__)

# Make type checkers happy without importing heavy deps at runtime
if TYPE_CHECKING:
    from .io.chunk import Chunk
    from .core.api import convert, open_flare, open_local_ancestry, open_rfmix, open_simu
    from .processing.constants import CHROM_SIZES, COORDINATES
    from .io.errors import BinaryFileNotFoundError
    from .readers.fb_read import read_fb
    from .processing.imputation import interpolate_array
    from .io.loci_bed import admix_to_bed_individual
    from .readers.read_flare import read_flare
    from .readers.read_msp import read_rfmix, extract_locus_ancestry
    from .readers.read_rfmix import read_rfmix_fb
    from .readers.read_simu import read_simu
    from .viz.tagore import plot_local_ancestry_tagore
    from .utils import (
        create_binaries,
        delete_files_or_directories,
        get_pops,
        get_prefixes,
        get_sample_names,
        set_gpu_environment,
    )
    from .viz.visualization import (
        generate_tagore_bed,
        plot_ancestry_by_chromosome,
        plot_global_ancestry,
        save_multi_format,
    )
    from .io.write_data import write_data
