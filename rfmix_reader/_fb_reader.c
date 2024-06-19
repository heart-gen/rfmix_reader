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
void read_fb_chunk(float *buff, uint64_t nrows, uint64_t ncols,
                   uint64_t row_start, uint64_t col_start, uint64_t row_end,
                   uint64_t col_end, int32_t *out, uint64_t *strides) {
  uint64_t r, c, ce;

  // Start from the row_start
  r = row_start;
  buff += r * ncols + col_start;
  
  // Process each row in the specific range
  while (r < row_end) {
    // Process each column in the specific range
    for (c = col_start; c < col_end;) {
      ce = MIN(c + 8, col_end);
      for (; c < ce; ++c) {
	out[(r - row_start) * strides[0] +
	    (c - col_start) * strides[1]] = (int32_t)buff[c];
      }
    }
    ++r;
    buff += ncols;
  }
}
