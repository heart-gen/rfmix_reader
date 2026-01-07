from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
from typing import Tuple

_GPU_ARRAY_AVAILABLE: bool | None = None
_GPU_DF_AVAILABLE: bool | None = None
_GPU_DASK_DF_AVAILABLE: bool | None = None

logger = logging.getLogger(__name__)


def _select_array_backend():
    """Return cupy if available; otherwise numpy with a warning on failure."""
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
    logger.warning(
        "CuPy unavailable or failed to initialize; "
        "falling back to NumPy CPU backend."
    )
    import numpy as np
    return np


def _select_dataframe_backend():
    """Return cudf if available; otherwise pandas with a warning on failure."""
    global _GPU_DF_AVAILABLE
    if _GPU_DF_AVAILABLE is False:
        import pandas as pd
        return pd

    module = None
    if importlib.util.find_spec("cudf") is not None:
        with contextlib.suppress(Exception):
            module = importlib.import_module("cudf")

    if module is not None:
        _GPU_DF_AVAILABLE = True
        return module

    _GPU_DF_AVAILABLE = False
    logger.warning(
        "cuDF unavailable or failed to initialize; "
        "falling back to pandas CPU backend."
    )
    import pandas as pd
    return pd


def _select_dask_dataframe_backend() -> Tuple[object, bool]:
    """Return dask_cudf module and True when available; else dask.dataframe."""
    global _GPU_DASK_DF_AVAILABLE
    if _GPU_DASK_DF_AVAILABLE is False:
        import dask.dataframe as dd
        return dd, False

    module = None
    if importlib.util.find_spec("dask_cudf") is not None:
        with contextlib.suppress(Exception):
            module = importlib.import_module("dask_cudf")

    if module is not None:
        _GPU_DASK_DF_AVAILABLE = True
        return module, True

    _GPU_DASK_DF_AVAILABLE = False
    logger.warning(
        "dask_cudf unavailable or failed to initialize; "
        "falling back to dask.dataframe CPU backend."
    )
    import dask.dataframe as dd
    return dd, False


def _configure_dask_backends() -> None:
    """Configure dask to use cudf/cupy when available."""
    from dask import config

    array_mod = _select_array_backend()
    df_mod = _select_dataframe_backend()

    config.set(
        {
            "array.backend": "cupy" if array_mod.__name__ == "cupy" else "numpy",
            "dataframe.backend": "cudf" if df_mod.__name__ == "cudf" else "pandas",
        }
    )
