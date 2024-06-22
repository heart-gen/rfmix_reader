"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from dask.delayed import delayed
from numpy import (
    ascontiguousarray,
    float32,
    memmap,
    int32
)
from dask.array import from_array, Array, concatenate

__all__ = ["read_fb"]

def read_fb(
        filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int
) -> Array:
    """
    Read and process data from a file in chunks, skipping the first
    2 rows (comments) and 4 columns (loci annotation).

    Parameters
    ----------
    filepath (str): Path to the binary file.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    row_chunk (int): Number of rows to process in each chunk.
    col_chunk (int): Number of columns to process in each chunk.

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
    chunks = []
    for ii in range(0, nrows, row_chunk):
        num_rows = min(row_chunk, nrows - ii)
        delayed_chunk = delayed(_read_chunk)(filepath, ii, num_rows, ncols)
        shape = (num_rows, col_chunk)
        dask_array_chunk = from_delayed(delayed_chunk,shape,float32)
        chunks.append(dask_array_chunk)
        
    # Concatenate all chunks
    X = concatenate(chunks, axis=0)
    return X


def _read_chunk(filepath, start_row, num_rows, num_cols):
    """
    Helper function to read a chunk of data from the binary file.

    Parameters
    ----------
    filepath (str): Path to the binary file.
    start_row (int): Starting row index for the chunk.
    num_rows (int): Number of rows in the chunk.
    num_cols (int): Number of columns in the chunk.

    Returns
    -------
    np.ndarray: The chunk of data read from the file.
    """
    base_size = float32().nbytes
    offset = start_row * num_cols * base_size
    buff = memmap(filepath, dtype=float32, mode="r",
                  offset=offset, shape=(num_rows, num_cols))
    return ascontiguousarray(buff, int32)
