import numpy as np
import dask.dataframe as dd
from sys import setrecursionlimit
from multiprocessing import cpu_count
from dask.array import (
    diff,
    where,
    Array,
    from_array
)

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, read_csv, concat
else:
    from pandas import DataFrame, read_csv, concat

__all__ = ["convert_loci"]

def convert_loci(loci: DataFrame, rf_q: DataFrame, admix: Array):
    # Column annotations
    sample_ids = _get_sample_names(rf_q)
    pops = _get_pops(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    # New Data
    annot_df = DataFrame(bed_format)
    dx = DataFrame(admix[idx,:].compute(),
                   columns=col_names)
    return None

df = DataFrame({
    'chromosome': ['chr1', 'chr1', 'chr1', 'chr2', 'chr2'],
    'physical_position': [1, 2, 3, 1, 2]
})
dask_matrix = from_array([
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0]
])
npops = 2

def _generate_bed(df: DataFrame, dask_matrix: Array, npops: int):
    # Check if the DataFrame and Dask array have the same number of rows
    assert df.shape[0] == dask_matrix.shape[0], "DataFrame and Dask array must have the same number of rows"
    # Convert the DataFrame to a Dask DataFrame
    parts = round(cpu_count()/2)
    ncols = dask_matrix.shape[1]
    ddf = dd.from_pandas(df.to_pandas(), npartitions=parts)    
    # Add each column of the Dask array to the DataFrame
    for i in range(ncols):
        ddf[f'data_{i}'] = dd.from_array(dask_matrix[:, i],
                                         chunksize=dask_matrix.chunksize[0])
    # Increase recursion limit
    setrecursionlimit(10000)

    # Loop through chromosomes
    results = []
    chromosomes = ddf['chromosome'].unique().compute()
    for chrom in chromosomes:
        chrom_group = ddf[ddf['chromosome'] == chrom]
        result = chrom_group.map_partitions(_process_chromosome,
                                            col_chunk=ncols,
                                            npops=npops)
        results.append(result)
    # Group by chromosome and apply the process_chromosome function
    bed_ddf = concat(results, axis=0)
    # Reset recursive limit to default value
    setrecursionlimit(1000)
    return bed_ddf


def _process_chromosome(
        group: dd.DataFrame, col_chunk: int, npops: int
) -> DataFrame:
    positions = group['physical_position'].to_dask_array(lengths=True)
    chromosome = group["chromosome"].unique().compute()[0]
    if len(chromosome) != 1:
        raise ValueError(f"Only one chromosome expected got: {len(chromosome)}")
    data_matrix = group.drop(["chromosome", "physical_position"], axis=1)\
                       .to_dask_array(lengths=True)
    change_indices = _find_intervals(data_matrix, npops)
    bed_records = []
    if change_indices:
        start = positions[0]
        for idx in change_indices:
            end = positions[idx + 1]
            bed_records.append([chromosome, start.compute(), end.compute()] +
                               data_matrix[idx].compute().tolist())
            start = end
        # Add the last interval
        bed_records.append([chromosome, start.compute(), positions[-1].compute()] +
                           data_matrix[-1].compute().tolist())    
    return DataFrame(bed_records, columns=['chromosome', 'start', 'end'] +
                     [f'data_{i}' for i in range(col_chunk)])


def _find_intervals(dask_matrix: Array, npops: int):
    ## This should be done per chromosome!
    num_cols = round(dask_matrix.shape[1] / npops) if npops == 2 else dask_matrix.shape[1]
    all_indices = set()    
    for col in range(num_cols):
        col_data = dask_matrix[:, col]
        diffs = diff(col_data)
        col_change_indices = where(diffs != 0)[0].compute()
        all_indices.update(col_change_indices)    
    return sorted(all_indices)


def _split_array(loci: DataFrame, admix: Array, npop: int) -> Array:
    chrom_idx = _find_chromosomes(loci)
    start_row = 0
    for end_row in chrom_idx:
        dask_matrix = admix[start_row:end_row+1]
        loci_chrom = loci.loc[start_row:end_row]
        loci_idx = _find_intervals(dask_matrix, npops)
        filtered_df = loci_chrom[loci_chrom["i"].isin(loci_idx)]
        physical_positions = filtered_df['physical_position'].to_arrow().tolist()
        start_row += end_row + 1
    return None


def _find_chromosomes(loci: DataFrame):
    from numpy import array, where
    matrix = loci.loc[:, "chromosome"].astype(str).to_numpy()
    return where(matrix[1:] != matrix[:-1])[0]


def _testing():
    from rfmix_reader import read_rfmix, create_binaries
    # prefix_path = "../examples/two_populations/out/"
    prefix_path = "/dcs05/lieber/hanlab/jbenjami/projects/"+\
        "localQTL_manuscript/local_ancestry_rfmix/_m/"
    create_binaries(prefix_path)
    loci, rf_q, admix = read_rfmix(prefix_path, verbose=True)


def _get_pops(rf_q: DataFrame):
    """
    Extract population names from an RFMix Q-matrix DataFrame.

    This function removes the 'sample_id' and 'chrom' columns from the input DataFrame
    and returns the remaining column names, which represent population names.

    Args:
        rf_q (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
            Expected to have 'sample_id' and 'chrom' columns, along with population columns.

    Returns:
        np.ndarray: An array of population names extracted from the column names.

    Example:
        If rf_q has columns ['sample_id', 'chrom', 'pop1', 'pop2', 'pop3'],
        this function will return ['pop1', 'pop2', 'pop3'].

    Note:
        This function assumes that all columns other than 'sample_id' and 'chrom'
        represent population names.
    """
    return rf_q.drop(["sample_id", "chrom"], axis=1).columns.values


def _get_sample_names(rf_q: DataFrame):
    """
    Extract unique sample IDs from an RFMix Q-matrix DataFrame and convert to Arrow array.

    This function retrieves unique values from the 'sample_id' column of the input DataFrame
    and converts them to a PyArrow array.

    Args:
        rf_q (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
            Expected to have a 'sample_id' column.

    Returns:
        pa.Array: A PyArrow array containing unique sample IDs.

    Example:
        If rf_q has a 'sample_id' column with values ['sample1', 'sample2', 'sample1', 'sample3'],
        this function will return a PyArrow array containing ['sample1', 'sample2', 'sample3'].

    Note:
        This function assumes that the 'sample_id' column exists in the input DataFrame.
        It uses PyArrow for efficient memory management and interoperability with other data processing libraries.
    """
    return rf_q.sample_id.unique().to_arrow()
