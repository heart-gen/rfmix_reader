"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from numpy import float32, memmap
from dask.array import from_array, Array

__all__ = ["read_fb"]

def read_fb(
        filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int
) -> Array:
    """
    Read and process data from a file in chunks, skipping the first
    2 rows (comments) and 4 columns (loci annotation).

    Parameters:
    filepath (str): Path to the binary file.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    row_chunk (int): Number of rows to process in each chunk.
    col_chunk (int): Number of columns to process in each chunk.

    Returns:
    dask.array: Concatenated array of processed data.
    """
    # Validate input parameters
    if row_chunk <= 0 or col_chunk <= 0:
        raise ValueError("row_chunk and col_chunk must be positive integers.")
    # Calculate row size and total size for memory mapping
    try:
        buff = memmap(f, dtype=float32, mode="r", offset=0,
                      shape=(nrows,ncols))
        X = from_array(buff, chunks=(row_chunk, col_chunk))
        buff._mmap.close()
        del buff
    except Exception as e:
        raise IOError(f"Error reading file: {e}")    
    assert isinstance(X, Array)
    return X

