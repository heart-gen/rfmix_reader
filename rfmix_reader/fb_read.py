"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from numpy import (
    ascontiguousarray,
    empty,
    float32,
    memmap,
    uint8,
    uint64,
    zeros,
)

__all__ = ["read_fb"]


def read_fb(filepath, nrows, ncols, npops, row_chunk, col_chunk):
    """
    Read and process data from a file in chunks, skipping the first
    2 rows (comments) and 4 columns (loci annotation).

    Parameters:
    filepath (str): Path to the file.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    npops (int): Number of populations for column processing.
    row_chunk (int): Number of rows to process in each chunk.
    col_chunk (int): Number of columns to process in each chunk.

    Returns:
    dask.array: Concatenated array of processed data.
    """
    from dask.delayed import delayed
    from dask.array import concatenate, from_delayed

    # Validate input parameters
    if nrows <= 2 or ncols <= 4:
        raise ValueError("Number of rows must be greater than 2 and number of columns must be greater than 4.")
    if row_chunk <= 0 or col_chunk <= 0:
        raise ValueError("row_chunk and col_chunk must be positive integers.")

    # Calculate row size and total size for memory mapping
    row_size = (ncols + 3) // 4
    size = nrows * row_size

    try:
        buff = memmap(filepath, uint8, "r", 3, shape=(size,))
    except Exception as e:
        raise IOError(f"Error reading file: {e}")
    
    row_start = 2 # Skip the first 2 rows
    column_chunks = []
    
    while row_start < nrows:
        row_end = min(row_start + row_chunk, nrows)
        col_start = 4 # Skip the first 4 columns
        row_chunks = []
        
        while col_start < ncols:
            col_end = min(col_start + col_chunk, ncols)
            x = delayed(_read_fb_chunk, None, True, None, False)(
                buff,
                nrows,
                ncols,
                npops,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            shape = (row_end - row_start, (col_end - col_start) // npops)
            row_chunks.append(from_delayed(x, shape, float32))
            col_start = col_end

        column_chunks.append(concatenate(row_chunks, 1, True))
        row_start = row_end
        
    return concatenate(column_chunks, 0, True)


def _read_fb_chunk(
        buff, nrows, ncols, npops, row_start, row_end, col_start, col_end
):
    """
    Read a chunk of data from the buffer and process it based on populations.

    Parameters:
    buff (memmap): Memory-mapped buffer containing the data.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    npops (int): Number of populations for column processing.
    row_start (int): Starting row index for the chunk.
    row_end (int): Ending row index for the chunk.
    col_start (int): Starting column index for the chunk.
    col_end (int): Ending column index for the chunk.

    Returns:
    dask.array: Processed array with adjacent columns summed for each population subset.
    """
    from fb_reader import ffi, lib

    base_type = uint8
    base_size = base_type().nbytes
    base_repr = "uint8_t"

    # Ensure the number of columns to be processed is even
    num_cols = col_end - col_start
    if num_cols % 2 != 0:
        raise ValueError("Number of columns to be summed must be even.")
    
    X = zeros((row_end - row_start, num_cols), base_type)
    assert X.flags.aligned

    strides = empty(2, uint64)
    strides[:] = X.strides
    strides //= base_size

    try:
        lib.read_fb_chunk(
            ffi.cast(f"{base_repr} *", buff.ctypes.data),
            nrows,
            ncols,
            row_start,
            col_start,
            row_end,
            col_end,
            ffi.cast(f"{base_repr} *", X.ctypes.data),
            ffi.cast("uint64_t *", strides.ctypes.data),
        )
    except Exception as e:
        raise IOError(f"Error reading data chunk: {e}")

    # Convert to contiguous array of type float32
    X = ascontiguousarray(X, float32)

    # Subset populations and sum adjacent columns
    return _subset_populations(X, npops)


def _subset_populations(X, npops):
        """
    Subset and process the input array X based on populations.

    Parameters:
    X (dask.array): Input array where columns represent data for different populations.
    npops (int): Number of populations.

    Returns:
    dask.array: Processed array with adjacent columns summed for each population subset.
    """
    from dask.array import concatenate
    pop_subset = []
    pop_start = 0
    ncols = X.shape[1]

    if ncols % npops != 0:
        raise ValueError("The number of columns in X must be divisible by npops.")
    
    while pop_start < npops:
        X0 = X[:, pop_start::npops] # Subset based on populations
        if X0.shape[1] % 2 != 0:
            raise ValueError("Number of columns must be even.")
        X0_summed = X0[:, ::2] + X0[:, 1::2] # Sum adjacent columns
        pop_subset.append(X0_summed)
        pop_start += 1

    return concatenate(pop_subset, 1, True)
