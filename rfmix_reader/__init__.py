from .chunk import Chunk
from .fb_read import read_fb
from .read_rfmix import read_rfmix
from ._utils import set_gpu_environment

__version__ = "0.1.5"

__all__ = [
    "Chunk",
    "__version__",
    "read_fb",
    "read_rfmix",
    "set_gpu_environment"
]
