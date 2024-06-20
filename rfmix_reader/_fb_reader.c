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
#include <string.h>

#define MIN(a, b) ((a > b) ? b : a)
#define UNROLL_FACTOR 8

// Function to read a chunk of the fb matrix
void read_fb_chunk(float *buff, uint64_t nrows, uint64_t ncols,
                   uint64_t row_start, uint64_t col_start, uint64_t row_end,
                   uint64_t col_end, int32_t *out) {
  uint64_t r, c;
  uint64_t row_size = ncols;
  uint64_t unrolled_cols = (col_end - col_start) / UNROLL_FACTOR * UNROLL_FACTOR;
  uint64_t remaining_cols = col_end - col_start - unrolled_cols;

  // Process each row in the specific range
  for (r = row_start; r < row_end; ++r) {
    // Process each column in unrolled chunks
    for (c = col_start; c < col_start + unrolled_cols; c += UNROLL_FACTOR) {
      out[(r - row_start) * row_size +
	  (c - col_start) + 0] = (int32_t)buff[r * row_size + c + 0];
      out[(r - row_start) * row_size +
	  (c - col_start) + 1] = (int32_t)buff[r * row_size + c + 1];
      out[(r - row_start) * row_size +
	  (c - col_start) + 2] = (int32_t)buff[r * row_size + c + 2];
      out[(r - row_start) * row_size +
	  (c - col_start) + 3] = (int32_t)buff[r * row_size + c + 3];
      out[(r - row_start) * row_size +
	  (c - col_start) + 4] = (int32_t)buff[r * row_size + c + 4];
      out[(r - row_start) * row_size +
	  (c - col_start) + 5] = (int32_t)buff[r * row_size + c + 5];
      out[(r - row_start) * row_size +
	  (c - col_start) + 6] = (int32_t)buff[r * row_size + c + 6];
      out[(r - row_start) * row_size +
	  (c - col_start) + 7] = (int32_t)buff[r * row_size + c + 7];
    }

    // Process remaining columns
    for (c = col_start + unrolled_cols; c < col_end; ++c) {
      out[(r - row_start) * row_size +
	  (c - col_start)] = (int32_t)buff[r * row_size + c];
    }
  }
}

/* int main() { */
/*     // Example usage */
/*     // Sample matrix (for demonstration purposes) */
/*     float matrix[4][4] = { */
/*         {0.0000, 1.0000, 0.0000, 1.0000}, */
/*         {1.0000, 0.0000, 1.0000, 0.0000}, */
/*         {0.0000, 1.0000, 0.0000, 1.0000}, */
/*         {1.0000, 0.0000, 1.0000, 0.0000} */
/*     }; */
/*     uint64_t nrows = 4; */
/*     uint64_t ncols = 4; */

/*     // Define start and end positions (for example purposes) */
/*     uint64_t row_start = 1, col_start = 1, row_end = 3, col_end = 3; */

/*     // Output buffer */
/*     int32_t out[2][2]; */
/*     memset(out, 0, sizeof(out)); */

/*     // Read the chunk */
/*     read_fb_chunk(&matrix[0][0], nrows, ncols, row_start, */
/* 		  col_start, row_end, col_end, &out[0][0]); */

/*     // Print the result */
/*     printf("Output:\n"); */
/*     for (uint64_t i = 0; i < row_end - row_start; ++i) { */
/*         for (uint64_t j = 0; j < col_end - col_start; ++j) { */
/*             printf("%d ", out[i][j]); */
/*         } */
/*         printf("\n"); */
/*     } */

/*     return 0; */
/* } */
