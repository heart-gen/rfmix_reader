/*
 * Adapted from the `_bed_reader.h` script in the `pandas-plink` package.
 * Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_reader.h
 * This is modified to handle a matrix of floating-point numbers.
 */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MIN(a, b) ((a > b) ? b : a)

// Function to read a chunk of the fb matrix
void read_fb_chunk(float *buff, uint64_t nrows, uint64_t ncols,
		   uint64_t row_start, uint64_t col_start, uint64_t row_end,
		   uint64_t col_end, float *out, uint64_t *strides) {
  uint32_t b; // Hold 32-bit integer representing the float
  uint64_t r, c, ce;
  uint64_t row_size; // in bytes
  float *buff_float = (float*)buff; // Cast buff to float pointer
  
  // Adjust for float size
  row_size = (ncols + 3) / 4 * sizeof(float); 

  r = row_start;
  buff_float += r * row_size + col_start / 4; // Access buff as float array

  while (r < row_end) {
    for (c = col_start; c < col_end;) {
      // Assuming buff is stored in little-endian format
      // (needs adjustment for big-endian)
      b = *((uint32_t*)buff_float + (c - col_start) / 4); // Read 4 floats at once

      // Extract bits using bitwise operations
      uint8_t b0 = b & 0xFF; // Get least significant byte (assuming 8-bit mantissa)
      uint8_t b1 = (b >> 8) & 0xFF; // Get second least significant byte
      uint8_t p0 = b0 ^ b1;
      uint8_t p1 = (b0 | b1) & b0;
      p1 <<= 1;
      p0 |= p1;

      ce = MIN(c + 4, col_end);
      for (; c < ce; ++c) {
        out[(r - row_start) * strides[0] +
	    (c - col_start) * strides[1]] = p0 & 3;
        p0 >>= 2;
      }
    }
    ++r;
    buff_float += row_size;
  }
}
