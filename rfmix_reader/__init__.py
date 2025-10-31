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
    "Chunk": ("._chunk", "Chunk"),
    "read_fb": ("._fb_read", "read_fb"),
    "read_simu": ("._read_simu", "read_simu"),
    "read_rfmix": ("._read_rfmix", "read_rfmix"),
    "read_flare": ("._read_flare", "read_flare"),
    "write_data": ("._write_data", "write_data"),
    "admix_to_bed_individual": ("._loci_bed", "admix_to_bed_individual"),
    "CHROM_SIZES": ("._constants", "CHROM_SIZES"),
    "COORDINATES": ("._constants", "COORDINATES"),
    "BinaryFileNotFoundError": ("._errorhandling", "BinaryFileNotFoundError"),
    "interpolate_array": ("._imputation", "interpolate_array"),
    "get_pops": ("._utils", "get_pops"),
    "get_prefixes": ("._utils", "get_prefixes"),
    "create_binaries": ("._utils", "create_binaries"),
    "get_sample_names": ("._utils", "get_sample_names"),
    "set_gpu_environment": ("._utils", "set_gpu_environment"),
    "delete_files_or_directories": ("._utils", "delete_files_or_directories"),
    "save_multi_format": ("._visualization", "save_multi_format"),
    "generate_tagore_bed": ("._visualization", "generate_tagore_bed"),
    "plot_global_ancestry": ("._visualization", "plot_global_ancestry"),
    "plot_ancestry_by_chromosome": ("._visualization", "plot_ancestry_by_chromosome"),
    "plot_local_ancestry_tagore": ("._tagore", "plot_local_ancestry_tagore"),
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
    from ._chunk import Chunk
    from ._fb_read import read_fb
    from ._read_simu import read_simu
    from ._read_rfmix import read_rfmix
    from ._read_flare import read_flare
    from ._write_data import write_data
    from ._loci_bed import admix_to_bed_individual
    from ._constants import CHROM_SIZES, COORDINATES
    from ._errorhandling import BinaryFileNotFoundError
    from ._imputation import interpolate_array
    from ._utils import (
        get_pops, get_prefixes, create_binaries, get_sample_names,
        set_gpu_environment, delete_files_or_directories,
    )
    from ._visualization import (
        save_multi_format, generate_tagore_bed,
        plot_global_ancestry, plot_ancestry_by_chromosome,
    )
    from ._tagore import plot_local_ancestry_tagore
