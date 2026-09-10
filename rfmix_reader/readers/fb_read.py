"""
Lazy reader for the raw float32 binary produced by :func:`create_binaries`.

The binary is a headerless, row-major ``float32`` dump of the data columns of
an RFMix ``.fb.tsv`` file (``nrows x ncols``).  Each dask block is produced by
memory-mapping a range of *whole rows* and slicing the requested columns, so
column chunks narrower than a row are read correctly.
"""
from __future__ import annotations

from os.path import getsize

import numpy as np
from dask.delayed import delayed
from dask.array import from_delayed, Array, concatenate

__all__ = ["read_fb"]

_ITEMSIZE = np.dtype(np.float32).itemsize


def read_fb(
        filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int
) -> Array:
    """
    Build a lazy ``(nrows, ncols)`` float32 dask array over a binary FB file.

    Parameters
    ----------
    filepath : str
        Path to the binary file written by :func:`create_binaries`.
    nrows, ncols : int
        Shape of the matrix stored in the file.
    row_chunk, col_chunk : int
        Block size along rows and columns.

    Returns
    -------
    dask.array.Array
        Lazy float32 array with the posterior probabilities as written.

    Raises
    ------
    ValueError
        If the chunk sizes are not positive, or if the file size does not
        match ``nrows * ncols * 4`` bytes (stale or mismatched binary).
    FileNotFoundError
        If the file does not exist.
    """
    if row_chunk <= 0 or col_chunk <= 0:
        raise ValueError("row_chunk and col_chunk must be positive integers.")

    expected = nrows * ncols * _ITEMSIZE
    actual = getsize(filepath)
    if actual != expected:
        raise ValueError(
            f"Binary file {filepath} has {actual} bytes but nrows*ncols*4 = "
            f"{expected} ({nrows} x {ncols}). The .bin is stale or was built "
            "from a different .fb.tsv; regenerate it with create_binaries()."
        )

    col_sx: list[Array] = []
    row_start = 0
    while row_start < nrows:
        row_end = min(row_start + row_chunk, nrows)
        col_start = 0
        row_sx: list[Array] = []
        while col_start < ncols:
            col_end = min(col_start + col_chunk, ncols)
            x = delayed(_read_chunk)(
                filepath, nrows, ncols, row_start, row_end, col_start, col_end,
            )
            shape = (row_end - row_start, col_end - col_start)
            row_sx.append(from_delayed(x, shape, dtype=np.float32))
            col_start = col_end
        col_sx.append(concatenate(row_sx, 1, True))
        row_start = row_end

    X = concatenate(col_sx, 0, True)
    assert isinstance(X, Array)
    return X


def _read_chunk(
        filepath, nrows, ncols, row_start, row_end, col_start, col_end
) -> np.ndarray:
    """
    Read rows ``[row_start, row_end)`` and columns ``[col_start, col_end)``.

    Whole rows are memory-mapped (they are contiguous on disk) and the column
    range is sliced from the map; the result is copied into a contiguous
    array so the mapping is released when the task returns.
    """
    if not (0 <= row_start < row_end <= nrows):
        raise ValueError(f"Row range [{row_start}, {row_end}) is outside [0, {nrows}).")
    if not (0 <= col_start < col_end <= ncols):
        raise ValueError(f"Column range [{col_start}, {col_end}) is outside [0, {ncols}).")

    offset = row_start * ncols * _ITEMSIZE
    buff = np.memmap(
        filepath, dtype=np.float32, mode="r", offset=offset,
        shape=(row_end - row_start, ncols),
    )
    return np.array(buff[:, col_start:col_end], dtype=np.float32)
