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

// Function to read a chunk of the fb matrix
void read_fb_chunk(float *buff, uint64_t nrows, uint64_t ncols,
                   uint64_t row_start, uint64_t col_start, uint64_t row_end,
                   uint64_t col_end, float *out, uint64_t *strides) {
  uint64_t r, c;

  // Start at the specific row and column
  buff += row_start * ncols + col_start;

  // Process each row in the specific range
  for (r = 0; r < (row_end - row_start); ++r) {
    // Process each column in the specific range
    for (c = 0; c < (col_end - col_start); ++c) {
      float value = buff[r * ncols + c];
      out[r * strides[0] + c * strides[1]] = value;
    }
  }
}
