/*
 * Adapted from the `_bed_reader.h` script in the `pandas-plink` package.
 * Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_reader.h
 */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MIN(a,b) ((a > b) ? b : a)

__global__
void read_fb_chunk_kernel(uint8_t *buff, uint64_t nrows, uint64_t ncols,
		          uint64_t row_start, uint64_t col_start, uint64_t row_end,
			  uint64_t col_end, uint8_t *out, uint64_t *strides,
			  uint64_t row_size) {
    int r = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int c = blockIdx.x * blockDim.x + threadIdx.x + col_start;

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

void read_fb_chunk(uint8_t *buff, uint64_t nrows, uint64_t ncols,
		   uint64_t row_start, uint64_t col_start, uint64_t row_end,
		   uint64_t col_end, uint8_t *out, uint64_t *strides) {
    uint64_t row_size = (ncols + 3) / 4;		   

    // Allocate GPU memory
    uint8_t *d_buff, *d_out;
    uint64_t *d_strides;
    cudaMalloc(&d_buff, nrows * row_size * sizeof(uint8_t));
    cudaMalloc(&d_out, (row_end - row_start) * (col_end - col_start) * sizeof(uint8_t));
    cudaMalloc(&d_strides, 2 * sizeof(uint64_t));

    // Copy data to GPU
    cudaMemcpy(d_buff, buff, nrows * row_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice)

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((col_end - col_start + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (row_end - row_start + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    read_fb_chunk_kernel<<<numBlocks, threadsPerBlock>>>(d_buff, nrows, ncols, row_start, col_start, row_end, col_end, d_out, d_strides, row_size);

    // Copy results back to host
    cudaMemcpy(out, d_out, (row_end - row_start) * (col_end - col_start) * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_buff);
    cudaFree(d_out);
    cudaFree(d_strides);
}
