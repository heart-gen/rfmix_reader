/*
 * Adapted from the `_bed_reader.h` script in the `pandas-plink` package.
 * Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_reader.h
 * This is modified to handle a matrix of floating-point numbers converted
 * to integer for reduced memory.
 */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define MIN(a, b) ((a > b) ? b : a)

// Function to read a chunk of the fb matrix
void read_fb_chunk(uint8_t *buff, uint64_t nrows, uint64_t ncols,
		   uint64_t row_start, uint64_t col_start, uint64_t row_end,
		   uint64_t col_end, uint8_t *out, uint64_t *strides) {
  uint8_t b;
  uint64_t r, c, ce;
  uint64_t row_size;
  
  // in bytes
  row_size = (ncols + 3) / 4 * sizeof(int); 

  r = row_start;
  buff += r * row_size + col_start / 4;

  while (r < row_end) {
    for (c = col_start; c < col_end;) {
      // Assuming buff is stored in little-endian format
      // (needs adjustment for big-endian)
      b = buff[(c - col_start) / 4];

      // Extract bits using bitwise operations
      uint8_t b0 = b & 0xFF;
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
    buff += row_size;
  }
}
