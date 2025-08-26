"""
Adapted from `_bed_read.py` script in the `pandas-plink` package.
Source: https://github.com/limix/pandas-plink/blob/main/pandas_plink/_bed_read.py
"""
from dask.delayed import delayed
from pysam import tabix_index, VariantFile
from dask.array import from_delayed, Array, concatenate
from numpy import (
    ascontiguousarray,
    float32,
    memmap,
    int32
)

__all__ = ["read_fb"]

def read_fb(
        filepath: str, nrows: int, ncols: int, row_chunk: int, col_chunk: int
) -> Array:
    """
    Read and process data from a file in chunks, skipping the first
    2 rows (comments) and 4 columns (loci annotation).

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    nrows : int
        Total number of rows in the dataset.
    ncols : int
        Total number of columns in the dataset.
    row_chunk : int
        Number of rows to process in each chunk.
    col_chunk : int
        Number of columns to process in each chunk.

    Returns
    -------
    dask.array: Concatenated array of processed data.

    Raises
    ------
    ValueError: If row_chunk or col_chunk is not a positive integer.
    FileNotFoundError: If the specified file does not exist.
    IOError: If there is an error reading the file.
    """
    # Validate input parameters
    if row_chunk <= 0 or col_chunk <= 0:
        raise ValueError("row_chunk and col_chunk must be positive integers.")

    # Calculate row size and total size for memory mapping
    col_sx: list[Array] = []
    row_start = 0
    while row_start < nrows:
        row_end = min(row_start + row_chunk, nrows)
        col_start = 0
        row_sx: list[Array] = []
        while col_start < ncols:
            col_end = min(col_start + col_chunk, ncols)
            x = delayed(_read_chunk)(
                filepath,
                nrows,
                ncols,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            shape = (row_end - row_start, col_end - col_start)
            row_sx.append(from_delayed(x, shape, dtype=float32))
            col_start = col_end
        col_sx.append(concatenate(row_sx, 1, True))
        row_start = row_end

    # Concatenate all chunks
    X = concatenate(col_sx, 0, True)
    assert isinstance(X, Array)
    return X


def _read_chunk(
        filepath, nrows, ncols, row_start, row_end, col_start, col_end
):
    """
    Helper function to read a chunk of data from the binary file.

    Parameters
    ----------
    filepath (str): Path to the binary file.
    nrows (int): Total number of rows in the dataset.
    ncols (int): Total number of columns in the dataset.
    row_start (int): Starting row index for the chunk.
    row_end (int): Ending row index for the chunk.
    col_start (int): Starting column index for the chunk.
    col_end (int): Ending column index for the chunk.

    Returns
    -------
    np.ndarray: The chunk of data read from the file.
    """
    base_size = float32().nbytes
    offset = (row_start * ncols + col_start) * base_size
    size = (row_end - row_start, col_end - col_start)

    buff = memmap(filepath, dtype=float32, mode="r",
                  offset=offset, shape=size)
    return ascontiguousarray(buff, int32)


def _load_vcf_data(vcf_file: str, chunk_size: int32 = 1_000_000
                   ) -> Iterator[DataFrame]:
    """
    Load VCF records from a BGZF compressed and tabix indexed VCF file in chunks
    using pysam and convert to DataFrames.

    Parameters
    ----------
    vcf_file : str
        Path to BGZF compressed VCF (.vcf.gz) with an associated .tbi index.
    chunk_size : int
        Number of records to include per chunk.

    Yields
    ------
    DataFrame
        DataFrame with 'chromosome' and 'physical_position' columns loaded chunk.
    """
    _create_tabix(vcf_file)
    vcf = VariantFile(vcf_file)
    records = []; count = 0
    fetch_iter = vcf.fetch()

    for rec in fetch_iter:
        records.append({'chromosome': rec.chrom, 'physical_position': rec.pos})
        count += 1
        if count % chunk_size == 0:
            yield DataFrame(records)
            records = []

    if records:
        yield DataFrame(records)
