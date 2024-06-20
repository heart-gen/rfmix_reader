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

// Function to read a chunk of the fb matrix
void read_fb_chunk(float *buff, uint64_t nrows, uint64_t ncols,
                   uint64_t row_start, uint64_t col_start, uint64_t row_end,
                   uint64_t col_end, float *out) {
  uint64_t r, c;
  
  buff += row_start * ncols + col_start;
    
  while (r < row_end) {
    for (c = col_start; c < col_start; ++c) {
      out[(r - row_start) * ncols +
	  (c - col_start)] = buff[r * ncols + c];
    }
    ++r;
    buff += ncols;
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
