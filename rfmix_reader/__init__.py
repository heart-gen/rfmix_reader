from pathlib import Path
from sys import version_info

from ._chunk import Chunk
from ._fb_read import read_fb
from ._read_rfmix import read_rfmix
from ._loci_bed import admix_to_bed_chromosome
from ._write_data import write_data, write_imputed
from ._errorhandling import BinaryFileNotFoundError
from ._imputation import interpolate_array, _expand_array
from ._utils import (
    get_prefixes,
    create_binaries,
    set_gpu_environment,
    delete_files_or_directories
)
from ._visualization import (
    save_multi_format,
    generate_tagore_bed,
    plot_global_ancestry,
    plot_ancestry_by_chromosome,
)

if version_info >= (3, 11):
    from tomllib import load
else:
    from toml import load

def get_version():
    """Read version dynamically from pyproject.toml"""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    #print(f"Searching for pyproject.toml at: {pyproject_path}")
    if pyproject_path.exists():
        with pyproject_path.open("rb") as f:
            return load(f)["tool"]["poetry"]["version"]
    return "0.0.0" # Default fallback

__version__ = get_version()

__all__ = [
    "Chunk",
    "read_fb",
    "write_data",
    "read_rfmix",
    "__version__",
    "get_prefixes",
    "_expand_array",
    "write_imputed",
    "create_binaries",
    "save_multi_format",
    "interpolate_array",
    "set_gpu_environment",
    "generate_tagore_bed",
    "plot_global_ancestry",
    "BinaryFileNotFoundError",
    "admix_to_bed_chromosome",
    "plot_ancestry_by_chromosome",
    "delete_files_or_directories",
]
