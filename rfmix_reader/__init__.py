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
    "read_fb", "read_simu", "read_rfmix", "read_flare",
    "write_data",
    "admix_to_bed_individual",
    "CHROM_SIZES", "COORDINATES",
    "BinaryFileNotFoundError",
    "interpolate_array",
    "get_pops", "get_prefixes", "create_binaries", "get_sample_names",
    "set_gpu_environment", "delete_files_or_directories",
    "save_multi_format", "generate_tagore_bed",
    "plot_global_ancestry", "plot_ancestry_by_chromosome",
    "plot_local_ancestry_tagore",
]

# Map public names for lazy loading
_lazy = {
    "Chunk": (".io.chunk", "Chunk"),
    "read_fb": (".fb_read", "read_fb"),
    "read_simu": (".read_simu", "read_simu"),
    "read_rfmix": (".read_rfmix", "read_rfmix"),
    "read_flare": (".read_flare", "read_flare"),
    "write_data": (".io.write_data", "write_data"),
    "admix_to_bed_individual": (".io.loci_bed", "admix_to_bed_individual"),
    "CHROM_SIZES": (".processing.constants", "CHROM_SIZES"),
    "COORDINATES": (".processing.constants", "COORDINATES"),
    "BinaryFileNotFoundError": (".io.errors", "BinaryFileNotFoundError"),
    "interpolate_array": (".processing.imputation", "interpolate_array"),
    "get_pops": (".utils", "get_pops"),
    "get_prefixes": (".utils", "get_prefixes"),
    "create_binaries": (".utils", "create_binaries"),
    "get_sample_names": (".utils", "get_sample_names"),
    "set_gpu_environment": (".utils", "set_gpu_environment"),
    "delete_files_or_directories": (".utils", "delete_files_or_directories"),
    "save_multi_format": (".viz.visualization", "save_multi_format"),
    "generate_tagore_bed": (".viz.visualization", "generate_tagore_bed"),
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
    from .processing.constants import CHROM_SIZES, COORDINATES
    from .io.errors import BinaryFileNotFoundError
    from .fb_read import read_fb
    from .processing.imputation import interpolate_array
    from .io.loci_bed import admix_to_bed_individual
    from .read_flare import read_flare
    from .read_rfmix import read_rfmix
    from .read_simu import read_simu
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
