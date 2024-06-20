"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from numpy import (
    ascontiguousarray,
    float32,
    memmap,
    uint64,
    zeros,
)

__all__ = ["read_fb"]

def read_fb(
        filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int
):
    """
    Read and process data from a file in chunks, skipping the first
    2 rows (comments) and 4 columns (loci annotation).

    Parameters:
    filepath (str): Path to the file.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    row_chunk (int): Number of rows to process in each chunk.
    col_chunk (int): Number of columns to process in each chunk.

    Returns:
    dask.array: Concatenated array of processed data.
    """
    from dask.delayed import delayed
    from dask.array import concatenate, from_delayed, Array

    # Validate input parameters
    if nrows <= 2 or ncols <= 4:
        raise ValueError("Number of rows must be greater than 2 and number of columns must be greater than 4.")
    if row_chunk <= 0 or col_chunk <= 0:
        raise ValueError("row_chunk and col_chunk must be positive integers.")

    # Calculate row size and total size for memory mapping
    try:
        size = nrows * ncols
        buff = memmap(filepath, dtype=float32,
                      mode="r", shape=(size,))
    except Exception as e:
        raise IOError(f"Error reading file: {e}")
    
    row_start = 2 # Skip the first 2 rows
    column_chunks: list[Array] = []
    while row_start < nrows:
        row_end = min(row_start + row_chunk, nrows)
        col_start = 4 # Skip the first 4 columns
        row_chunks: list[Array] = []
        while col_start < ncols:
            col_end = min(col_start + col_chunk, ncols)
            
            x = delayed(_read_fb_chunk, None, True, None, False)(
                buff,
                nrows,
                ncols,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            
            shape = (row_end - row_start, col_end - col_start)
            row_chunks.append(from_delayed(x, shape, float32))
            col_start = col_end

        column_chunks.append(concatenate(row_chunks, 1, True))
        row_start = row_end
    X = concatenate(column_chunks, 0, True)
    assert isinstance(X, Array)
    return X


def _read_fb_chunk(
        buff, nrows, ncols, row_start, row_end, col_start, col_end
):
    """
    Read a chunk of data from the buffer and process it based on populations.

    Parameters:
    buff (memmap): Memory-mapped buffer containing the data.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.    
    row_start (int): Starting row index for the chunk.
    row_end (int): Ending row index for the chunk.
    col_start (int): Starting column index for the chunk.
    col_end (int): Ending column index for the chunk.

    Returns:
    dask.array: Processed array with adjacent columns summed for each population subset.
    """
    from .fb_reader import ffi, lib
    
    # Ensure the number of columns to be processed is even
    num_cols = col_end - col_start
    if num_cols % 2 != 0:
        raise ValueError("Number of columns must be even.")
    
    X = zeros((row_end - row_start, num_cols), float32)
    assert X.flags.aligned
    
    try:
        lib.read_fb_chunk(
            ffi.cast(f"float *", buff.ctypes.data),
            nrows,
            ncols,
            row_start,
            col_start,
            row_end,
            col_end,
            ffi.cast("float *", X.ctypes.data),
        )
    except Exception as e:
        raise IOError(f"Error reading data chunk: {e}")
    return ascontiguousarray(X, float32)
