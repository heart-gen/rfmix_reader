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
  uint64_t r, c, ce;
  uint64_t row_size = ncols * sizeof(float); 

  // Start at the specific row and column
  float *float_buff = (float *)buff;
  float_buff += row_start * ncols + col_start;

  // Process each row in the specific range
  for (r = row_start; r < row_end; ++r) {
    // Process each column in the specific range
    for (c = col_start; c < col_end;) {
      float value = float_buff[(r - row_start) * ncols + (c - col_start)];
      uint8_t int_value = (uint8_t)roundf(value);
      ce = MIN(c + 1, col_end);
      for (; c < ce; ++c) {
	out[(r - row_start) * strides[0] +
	    (c - col_start) * strides[1]] = int_value;
      }
    }
  }
}
