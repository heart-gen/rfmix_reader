"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
This script has been edited to increase speed with CUDA GPUs.
"""
from numpy import (
    float32,
    memmap,
    uint8,
    zeros,
)

from numba import cuda
from dask.delayed import delayed
from dask.array import concatenate, from_delayed

__all__ = ["read_fb"]


def read_fb(filepath, nrows, ncols, row_chunk, col_chunk):
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
                row_start,
                row_end,
                col_start,
                col_end,
            )
            shape = (row_end - row_start, (col_end - col_start))
            row_chunks.append(from_delayed(x, shape, float32))
            col_start = col_end

        column_chunks.append(concatenate(row_chunks, 1, True))
        row_start = row_end
        
    return concatenate(column_chunks, 0, True)

@cuda.jit
def read_fb_chunk_kernel(d_buff, d_out, nrows, ncols, row_start,
                         col_start, row_end, col_end, row_size):
    r = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y + row_start
    c = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x + col_start

    if r < row_end and c < col_end:
        buff_index = r * row_size + c // 4
        b = d_buff[buff_index]
        b0 = b & 0x55
        b1 = (b & 0xAA) >> 1
        p0 = b0 ^ b1
        p1 = (b0 | b1) & b0
        p1 <<= 1
        p0 |= p1
        ce = min(c + 4, col_end)

        while c < ce:
            d_out[r - row_start, c - col_start] = p0 & 3
            p0 >>= 2
            c += 1


def _read_fb_chunk(buff, nrows, ncols, row_start, row_end, col_start, col_end):
    """
    Read a chunk of data from the buffer and process it using CUDA.

    Parameters:
    buff (np.memmap): Memory-mapped buffer containing the data.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.    
    row_start (int): Starting row index for the chunk.
    row_end (int): Ending row index for the chunk.
    col_start (int): Starting column index for the chunk.
    col_end (int): Ending column index for the chunk.

    Returns:
    np.ndarray: Array of data.
    """
    # Ensure the number of columns to be processed is even
    num_cols = col_end - col_start
    if num_cols % 2 != 0:
        raise ValueError("Number of columns must be even.")
    
    X = zeros((row_end - row_start, num_cols), dtype=uint8)
    row_size = (ncols + 3) // 4
    
    # Copy data to GPU
    d_buff = cuda.to_device(buff)
    d_out = cuda.to_device(X)

    # Define block and grid sizes
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (num_cols + threads_per_block[0] - 1) // threads_per_block[0],
        (row_end - row_start + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch the kernel
    read_fb_chunk_kernel[blocks_per_grid, threads_per_block](
        d_buff, d_out, nrows, ncols, row_start, col_start,
        row_end, col_end, row_size
    )

    # Copy the results back to host memory
    results = d_out.copy_to_host()
    return results.astype(float32)
