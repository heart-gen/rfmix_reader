from pathlib import Path
from sys import version_info

from ._chunk import Chunk
from ._fb_read import read_fb
from ._read_rfmix import read_rfmix
from ._write_data import write_data, write_imputed
from ._errorhandling import BinaryFileNotFoundError
from ._imputation import interpolate_array, _expand_array
from ._loci_bed import (
    generate_tagore_bed,
    admix_to_bed_chromosome,
)
from ._utils import (
    set_gpu_environment,
    delete_files_or_directories,
    get_prefixes, create_binaries
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
    "write_imputed",
    "set_gpu_environment",
    "generate_tagore_bed",
    "BinaryFileNotFoundError",
    "admix_to_bed_chromosome",
    "delete_files_or_directories",
    "get_prefixes", "create_binaries",
    "interpolate_array", "_expand_array",
]
