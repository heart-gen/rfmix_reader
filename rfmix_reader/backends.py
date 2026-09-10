"""
Compute-layer backend selection.

Parsers are plain NumPy; GPU acceleration is opt-in for the compute layer
(imputation, plotting helpers) through CuPy when it is importable.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging

_GPU_ARRAY_AVAILABLE: bool | None = None

logger = logging.getLogger(__name__)


def _select_array_backend():
    """Return cupy if importable; otherwise numpy."""
    global _GPU_ARRAY_AVAILABLE
    if _GPU_ARRAY_AVAILABLE is False:
        import numpy as np
        return np

    module = None
    if importlib.util.find_spec("cupy") is not None:
        with contextlib.suppress(Exception):
            module = importlib.import_module("cupy")

    if module is not None:
        _GPU_ARRAY_AVAILABLE = True
        return module

    _GPU_ARRAY_AVAILABLE = False
    logger.debug("CuPy unavailable or failed to initialize; using the NumPy backend.")
    import numpy as np
    return np


def use_gpu() -> bool:
    """``True`` when the CuPy backend is selected."""
    return _select_array_backend().__name__ == "cupy"


def _configure_dask_backends() -> None:
    """Configure dask to use cupy for arrays when available."""
    from dask import config

    config.set({"array.backend": "cupy" if use_gpu() else "numpy"})


def describe_gpus() -> list[dict]:
    """
    Properties of the CUDA devices visible to PyTorch (name, memory in GB,
    capability); an empty list when PyTorch or CUDA is unavailable.
    """
    try:
        from torch.cuda import device_count, get_device_properties
    except ImportError:
        logger.debug("PyTorch is not installed; GPU information is unavailable.")
        return []
    devices = []
    for num in range(device_count()):
        props = get_device_properties(num)
        devices.append({
            "index": num, "name": props.name,
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "capability": f"{props.major}.{props.minor}",
        })
        logger.info("GPU %d: %s, %.2f GB, CUDA capability %s", num, props.name,
                    props.total_memory / (1024 ** 3), f"{props.major}.{props.minor}")
    return devices
