rfmix\_reader.read\_fb
======================

.. function:: read_fb(filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int) -> Array

    Read and process data from a file in chunks, skipping the first 2 rows (comments) and 4 columns (loci annotation).

    :param filepath: Path to the binary file.
    :type filepath: str
    :param nrows: Total number of rows in the dataset.
    :type nrows: int
    :param ncols: Total number of columns in the dataset.
    :type ncols: int
    :param row_chunk: Number of rows to process in each chunk.
    :type row_chunk: int
    :param col_chunk: Number of columns to process in each chunk.
    :type col_chunk: int

    :returns: Concatenated array of processed data.
    :rtype: dask.array.Array

    :raises ValueError: If row_chunk or col_chunk is not a positive integer.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises IOError: If there is an error reading the file.

    Example
    -------
    ::

        array = read_fb("datafile.bin", 1000, 1000, 100, 100)
