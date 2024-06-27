"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from dask.delayed import delayed
from dask.array import from_delayed, Array, concatenate
from numpy import (
    ascontiguousarray,
    float32,
    memmap,
    int32
)

__all__ = ["read_fb"]

def read_fb(
        filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int
) -> Array:
    """
    Read and process data from a file in chunks, skipping the first
    2 rows (comments) and 4 columns (loci annotation).

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    nrows : int
        Total number of rows in the dataset.
    ncols : int
        Total number of columns in the dataset.
    row_chunk : int
        Number of rows to process in each chunk.
    col_chunk : int
        Number of columns to process in each chunk.

    Returns
    -------
    dask.array: Concatenated array of processed data.

    Raises
    ------
    ValueError: If row_chunk or col_chunk is not a positive integer.
    FileNotFoundError: If the specified file does not exist.
    IOError: If there is an error reading the file.
    """
    # Validate input parameters
    if row_chunk <= 0 or col_chunk <= 0:
        raise ValueError("row_chunk and col_chunk must be positive integers.")
    
    # Calculate row size and total size for memory mapping
    col_sx: list[Array] = []
    row_start = 0
    while row_start < nrows:
        row_end = min(row_start + row_chunk, nrows)
        col_start = 0
        row_sx: list[Array] = []
        while col_start < ncols:
            col_end = min(col_start + col_chunk, ncols)
            x = delayed(_read_chunk)(
                filepath,
                nrows,
                ncols,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            shape = (row_end - row_start, col_end - col_start)
            row_sx.append(from_delayed(x, shape, dtype=float32))
            col_start = col_end
        col_sx.append(concatenate(row_sx, 1, True))
        row_start = row_end
        
    # Concatenate all chunks
    X = concatenate(col_sx, 0, True)
    assert isinstance(X, Array)
    return X


def _read_chunk(
        filepath, nrows, ncols, row_start, row_end, col_start, col_end
):
    """
    Helper function to read a chunk of data from the binary file.

    Parameters
    ----------
    filepath (str): Path to the binary file.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    row_start (int): Starting row index for the chunk.
    row_end (int): Ending row index for the chunk.
    col_start (int): Starting column index for the chunk.
    col_end (int): Ending column index for the chunk.

    Returns
    -------
    np.ndarray: The chunk of data read from the file.
    """
    base_size = float32().nbytes
    offset = (row_start * ncols + col_start) * base_size
    size = (row_end - row_start, col_end - col_start)
    
    buff = memmap(filepath, dtype=float32, mode="r",
                  offset=offset, shape=size)
    return ascontiguousarray(buff, int32)
    
