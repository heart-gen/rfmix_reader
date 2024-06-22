"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from dask.delayed import delayed
from numpy import float32, memmap, array
from dask.array import from_array, Array, concatenate

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
    chunks = []
    for ii in range(0, nrows, row_chunk):
        num_rows = min(row_chunk, nrows - ii)
        delayed_chunk = delayed(_read_chunk, None, True, None, False)(
            filepath, ii, num_rows, ncols)
        dask_array_chunk = from_array(delayed_chunk,
                                      chunks=(num_rows, col_chunk))
        chunks.append(dask_array_chunk)
    # Concategnate all chunks
    X = concatenate(chunks, axis=0, True)
    assert isinstance(X, Array)
    return X


def _read_chunk(filepath, start_row, num_rows, num_cols):
    """
    Read a chunk of data from the binary file.

    Parameters:
    filepath (str): Path to the binary file.
    start_row (int): Starting row index for the chunk.
    num_rows (int): Number of rows in the chunk.
    num_cols (int): Number of columns in the chunk.

    Returns:
    np.ndarray: Chunk of data.
    """
    with memmap(filepath, dtype=float32, mode="r",
                offset=start_row * num_cols * float32().itemsize,
                shape=(num_rows, num_cols)) as buff:
        return array(buff)
