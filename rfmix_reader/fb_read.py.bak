"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
This script has been edited to increase speed with CUDA GPUs.
"""
import cupy as cp
import numpy as np
from cupyx import jit
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
        buff = np.memmap(filepath, np.uint8, "r", 3, shape=(size,))
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
            row_chunks.append(from_delayed(x, shape, np.float32))
            col_start = col_end

        column_chunks.append(concatenate(row_chunks, 1, True))
        row_start = row_end
        
    return concatenate(column_chunks, 0, True)


def _read_fb_chunk(
        buff, nrows, ncols, row_start, row_end, col_start, col_end
):
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
    
    X = cp.zeros((row_end - row_start, num_cols), dtype=cp.uint8)
    strides = np.array([X.strides[0] // X.itemsize, X.strides[1] // X.itemsize],
                       dtype=np.uint64)
    row_size = (ncols + 3) // 4
    
    # Copy data to GPU
    d_buff = cp.asarray(buff)
    d_out = cp.zeros_like(X)

    # Define block and grid sizes
    threads_per_block = (32, 16)
    blocks_per_grid = (
        (num_cols + threads_per_block[0] - 1) // threads_per_block[0],
        (row_end - row_start + threads_per_block[1] - 1) // threads_per_block[1]
    )

    # Launch the kernel
    read_fb_chunk_kernel((blocks_per_grid, threads_per_block),
        d_buff, nrows, ncols, row_start, col_start, row_end, col_end,
        d_out, cp.asarray(strides), row_size
    )

    # Copy the results back to host memory
    results = cp.asnumpy(d_out)
    return results.astype(np.float32)


@jit.rawkernel()
def read_fb_chunk_kernel(d_buff, nrows, ncols, row_start, col_start,
                         row_end, col_end, d_out, d_strides, row_size):
    """
    CUDA kernel function to read a chunk of data and process it.

    Parameters:
    d_buff (cupy.ndarray): Device buffer containing the data.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    row_start (int): Starting row index for the chunk.
    row_end (int): Ending row index for the chunk.
    col_start (int): Starting column index for the chunk.
    col_end (int): Ending column index for the chunk.
    d_out (cupy.ndarray): Output buffer to store the processed data.
    d_strides (cupy.ndarray): Strides for the output buffer.
    row_size (int): Row size in the buffer.
    """
    r = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y + row_start
    c = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x + col_start

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
            d_out[(r - row_start) * d_strides[0] +
                  (c - col_start) * d_strides[1]] = p0 & 3
            p0 >>= 2
            c += 1


source_code = r'''
extern "C"{
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define MIN(a,b) ((a > b) ? b : a)

__global__ void read_fb_chunk_kernel(uint8_t *buff, uint64_t nrows, uint64_t ncols,
		          uint64_t row_start, uint64_t col_start, uint64_t row_end,
			  uint64_t col_end, uint8_t *out, uint64_t *strides,
			  uint64_t row_size) {
    // Thread indices within the block
    int r = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int c = blockIdx.x * blockDim.x + threadIdx.x + col_start;

    // Check if within valid data range
    if (r < row_end && c < col_end) {
        uint64_t buff_index = r * row_size + c / 4;
	char b = buff[buff_index];
	char b0 = b & 0x55;
	char b1 = (b & 0xAA) >> 1;
	char p0 = b0 ^ b1;
	char p1 = (b0 | b1) & b0;
	p1 <<= 1;
	p0 |= p1;
	uint64_t ce = MIN(c + 4, col_end);

        for (; c < ce; ++c) {
	    out[(r - row_start) * strides[0] + (c - col_start) * strides[1]] = p0 & 3;
	    p0 >>= 2;
	}
    }
}
}
'''
