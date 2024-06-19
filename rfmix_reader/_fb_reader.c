/*
 * Adapted from the `_bed_reader.h` script in the `pandas-plink` package.
 * Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_reader.h
 * This is modified to handle a matrix of floating-point numbers.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Function to read a chunk of the fb matrix
void read_fb_chunk(float *matrix, uint64_t nrows, uint64_t ncols,
		   uint64_t row_start, uint64_t col_start, uint64_t row_end,
		   uint64_t col_end, float *out, uint64_t *strides) {
  uint64_t r, c;
  uint64_t row_size = ncols;

  // Start at the specific row and column
  matrix += row_start * row_size + col_start;

  // Process each row in the specific range
  #pragma omp parallel for private(c)
  for (r = row_start; r < row_end; ++r) {
    // Process each column in the specific range
    #pragma omp simd
    for (c = col_start; c < col_end; ++c) {
      // Read the value from the matrix
      float value = matrix[(r - row_start) * row_size + (c - col_start)];

      // Copy the value to the output buffer
      out[(r - row_start) * strides[0] + (c - col_start) * strides[1]] = value;
    }
  }
}

/* int main() { */
/*     // Example usage of read_fb_chunk */
/*     // Sample matrix (for demonstration purposes) */
/*     float matrix[4][4] = { */
/*         {1.0000, 0.0000, 0.0000, 1.0000}, */
/*         {1.0000, 0.0000, 1.0000, 0.0000}, */
/*         {0.0000, 1.0000, 1.0000, 0.0000}, */
/*         {0.0000, 1.0000, 0.0000, 1.0000} */
/*     }; */
/*     uint64_t nrows = 4; */
/*     uint64_t ncols = 4; */

/*     // Define start and end positions (for example purposes) */
/*     uint64_t row_start = 1, col_start = 1, row_end = 3, col_end = 3; */

/*     // Output buffer */
/*     float out[2][2]; */
/*     memset(out, 0, sizeof(out)); */

/*     // Strides for the output buffer */
/*     uint64_t strides[2] = {2, 1}; */

/*     // Read the chunk */
/*     read_fb_chunk(&matrix[0][0], nrows, ncols, row_start, col_start, */
/* 		  row_end, col_end, &out[0][0], strides); */

/*     // Print the result */
/*     printf("Output:\n"); */
/*     for (uint64_t i = 0; i < row_end - row_start; ++i) { */
/*         for (uint64_t j = 0; j < col_end - col_start; ++j) { */
/*             printf("%.4f ", out[i][j]); */
/*         } */
/*         printf("\n"); */
/*     } */

/*     return 0; */
/* } */
