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

try:
    from torch.cuda import is_available
except ModuleNotFoundError:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    import cupy as cp
    from cupyx import jit

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
    # if is_available():
    #     # Use GPU to read and process the chunk
    #     num_cols = col_end - col_start
    #     if num_cols % 2 != 0:
    #         raise ValueError("Number of columns must be even.")    
    #     X = cp.zeros((row_end - row_start, num_cols), dtype=cp.uint8)
    #     strides = np.array([X.strides[0] // X.itemsize, X.strides[1] // X.itemsize],
    #                        dtype=np.uint64)
    #     row_size = (ncols + 3) // 4    
    #     # Copy data to GPU
    #     d_buff = cp.asarray(buff)
    #     d_out = cp.zeros_like(X)
    #     # Define block and grid sizes
    #     threads_per_block = (16, 16)
    #     blocks_per_grid = (
    #         (col_end - col_start + threads_per_block[0] - 1) // threads_per_block[0],
    #         (row_end - row_start + threads_per_block[1] - 1) // threads_per_block[1]
    #     )
    #     # Launch the kernel
    #     read_fb_chunk_kernel((blocks_per_grid, threads_per_block),
    #         d_buff, nrows, ncols, row_start, col_start, row_end, col_end,
    #         d_out, cp.asarray(strides), row_size
    #     )
    #     # Copy the results back to host memory
    #     results = cp.asnumpy(d_out)
    #     return results.astype(float32)
    from .fb_reader import ffi, lib
    # Use C program
    base_type = uint8
    base_size = base_type().nbytes
    base_repr = "uint8_t"
    # Ensure the number of columns to be processed is even
    num_cols = col_end - col_start
    if num_cols % 2 != 0:
        raise ValueError("Number of columns must be even.")    
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
    return ascontiguousarray(X, float32)


if is_available():
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
