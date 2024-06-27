from tqdm import tqdm
from numpy import full
from typing import List
from os import makedirs
from os.path import join
import dask.dataframe as dd
from multiprocessing import cpu_count
from dask.array import (
    diff,
    where,
    Array,
    array,
    from_array,
    concatenate
)

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, concat
else:
    from pandas import DataFrame, concat

__all__ = ["loci_to_bed"]

def _testing():
    from rfmix_reader import read_rfmix, create_binaries
    # prefix_path = "../examples/two_populations/out/"
    prefix_path = "/dcs05/lieber/hanlab/jbenjami/projects/"+\
        "localQTL_manuscript/local_ancestry_rfmix/_m/"
    #create_binaries(prefix_path)
    loci, rf_q, admix = read_rfmix(prefix_path)
    loci_to_bed(loci, rf_q, admix)
    return None


def loci_to_bed(loci: DataFrame, rf_q: DataFrame, admix: Array) -> None:
    # Column annotations
    sample_ids = _get_sample_names(rf_q); pops = _get_pops(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    # Generate BED dataframe
    _generate_bed(loci, admix, len(pops), col_names)
    return None


def _generate_bed(
        df: DataFrame, dask_matrix: Array, npops: int,
        col_names: List[str], output_dir: str = "output",
        output_file: str = "bed_results.hdf",
        verbose: bool = True
) -> None:
    # Check if the DataFrame and Dask array have the same number of rows
    assert df.shape[0] == dask_matrix.shape[0], "DataFrame and Dask array must have the same number of rows"
    # Convert the DataFrame to a Dask DataFrame
    parts = cpu_count()
    ncols = dask_matrix.shape[1]
    if is_available() and isinstance(df, DataFrame):
        ddf = dd.from_pandas(df.to_pandas(), npartitions=parts)
    else:
        ddf = dd.from_pandas(df, npartitions=parts)
    # Add each column of the Dask array to the DataFrame
    dask_df = dd.from_dask_array(dask_matrix, columns=col_names)
    ddf = dd.concat([ddf, dask_df], axis=1)
    del dask_df    
    # Loop through results
    chromosomes = ddf['chromosome'].unique().compute()
    makedirs(output_dir, exist_ok=True)
    out_file = join(output_dir, output_file)
    for chrom in tqdm(sorted(chromosomes), desc="Processing Chromosomes",
                      disable=not verbose):
        chrom_group = ddf[ddf['chromosome'] == chrom]
        bed_records = _process_chromosome(chrom_group, npops, col_names)
        # Write results to Parquet file
        bed_records.to_hdf(out_file, key="bed_records", mode="a", complevel=9,
                           format="table", data_columns=True, index=False)
        # Clear memory
        del bed_records, chrom_group
    if verbose:
        print(f"Results written to {out_file}")
    return None


def _process_chromosome(
        group: dd.DataFrame, npops: int, col_names: List[str]
) -> DataFrame:
    """
    Process genetic data for a single chromosome to identify ancestry
    intervals.

    This function takes a Dask DataFrame representing genetic data for
    a single chromosome, identifies intervals where ancestry changes,
    and creates a BED-like format DataFrame with these intervals.

    Parameters
    ----------
    group (dd.DataFrame): A Dask DataFrame containing genetic data for
                          a single chromosome.
    npops (int): The number of ancestral populations.
    col_names (List[str]): Names of the ancestry data columns.

    Returns
    -------
    DataFrame: A DataFrame (pandas or cudf) in BED-like format with columns:
        'chromosome', 'start', 'end', and ancestry data columns.

    Raises
    ------
    ValueError: If the input group contains data for more than one chromosome.

    Notes
    -----
    - This function assumes that the input group is sorted by physical position.
    - It uses the `_find_intervals` function to identify change points in ancestry.
    - The output DataFrame includes one row for each interval where ancestry
      is constant.

    Example
    -------
    Input group (simplified):
    chromosome | physical_position | pop1 | pop2
    1          | 100               | 1    | 0
    1          | 200               | 1    | 0
    1          | 300               | 0    | 1
    1          | 400               | 0    | 1

    Output Dask DataFrame:
    chromosome | start | end | pop1 | pop2
    1          | 100   | 200 | 1    | 0
    1          | 300   | 400 | 0    | 1
    """ #L57
    # Convert 'physical_position' column to a Dask array
    positions = group['physical_position'].to_dask_array(lengths=True)
    # Ensure the DataFrame contains data for only one chromosome
    chromosome = group["chromosome"].unique()
    if len(chromosome) != 1:
        raise ValueError(f"Only one chromosome expected got: {len(chromosome)}")
    # Convert the data matrix to a Dask array, excluding 'chromosome' 
    # and 'physical_position' columns
    data_matrix = group[col_names].to_dask_array(lengths=True)
    # Find indices where genetic changes occur
    change_indices = array(_find_intervals(data_matrix, npops))
    # Compute chromosome value once
    chromosome_value = chromosome.compute()[0]
    # Create arrays for each column
    # Create Dask DataFrame
    bed_records = _create_bed_records(chromosome_value, positions,
                                      data_matrix, change_indices)
    cnames = ['chromosome', 'start', 'end'] + col_names
    return dd.from_dask_array(bed_records, columns=cnames)


def _create_bed_records(chromosome_value, positions, data_matrix, change_indices):
    # Create start indices
    start_indices = concatenate([array([0]), change_indices[:-1] + 1])
    # Create arrays for each column
    chrom_column = full(change_indices.shape, chromosome_value)
    start_column = positions[start_indices]
    end_column = positions[change_indices]
    data_columns = data_matrix[change_indices]
    # Add the last interval
    last_start = positions[change_indices[-1] + 1]
    last_end = positions[-1]
    last_data = data_matrix[-1]
    # Concatenate all data
    chrom_column = concatenate([chrom_column, array([chromosome_value])])
    start_column = concatenate([start_column, array([last_start])])
    end_column = concatenate([end_column, array([last_end])])
    data_columns = concatenate([data_columns, last_data.reshape(1, -1)])
    # Combine all columns
    return concatenate([chrom_column[:, None], start_column[:, None],
                        end_column[:, None], data_columns], axis=1)


def _find_intervals(data_matrix: Array, npops: int) -> List[int]:
    """
    Find the indices where changes occur in a Dask array representing
    genetic data.

    This function identifies positions where the genetic ancestry changes
    across samples for a single chromosome. It processes the data column
    by column and aggregates all unique change positions.

    Parameters
    ----------
    data_matrix (dask.array.Array): A 2D Dask array representing genetic
        ancestry data. Each row corresponds to a position along the 
        chromosome, and each column (or group of columns for npops > 2) 
        represents a sample.
    npops (int): The number of ancestral populations. This affects how
        columns are grouped and processed.

    Returns
    -------
    List[int]: A sorted list of indices where changes in genetic ancestry occur.

    Notes
    -----
    - For npops == 2, we drop half as they should be 1-pop. This is to
      speed up computation.
    - For npops > 2, each column is treated independently (conservative).
    - The function uses Dask for efficient processing of large datasets.

    Example
    -------
    Given a Dask matrix representing ancestry along a chromosome for 3 
    samples and 2 populations:
        [[1, 0, 1, 0, 1, 0],
         [1, 0, 1, 0, 0, 1],
         [1, 0, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1]]

    _find_intervals(data_matrix, npops=2) might return [1, 2]

    Raises
    ------
    ValueError: If the input matrix dimensions are inconsistent with npops.

    Performance Considerations
    --------------------------
    - This function may be computationally intensive for large matrices.
    - It performs computations lazily until the final `compute()` call.
    """
    # Validate input dimensions
    if data_matrix.shape[1] % npops != 0:
        raise ValueError(f"Matrix column count ({data_matrix.shape[1]}) is not divisible by npops ({npops})")
    # Determine the number of columns to process
    num_cols = data_matrix.shape[1] // npops if npops == 2 else data_matrix.shape[1]
    # Vectorized diff operation across all relevant columns
    diffs = diff(data_matrix[:, :num_cols], axis=0)    
    # Find non-zero elements across all columns at once
    changes = where(diffs != 0)    
    # Compute only once and convert to set for faster operations
    return sorted(set(changes[0].compute()))


def _get_pops(rf_q: DataFrame):
    """
    Extract population names from an RFMix Q-matrix DataFrame.

    This function removes the 'sample_id' and 'chrom' columns from
    the input DataFrame and returns the remaining column names, which
    represent population names.

    Parameters
    ----------
    rf_q (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
        Expected to have 'sample_id' and 'chrom' columns, along with
        population columns.

    Returns
    -------
    np.ndarray: An array of population names extracted from the column names.

    Example
    -------
    If rf_q has columns ['sample_id', 'chrom', 'pop1', 'pop2', 'pop3'],
    this function will return ['pop1', 'pop2', 'pop3'].

    Note
    ----
    This function assumes that all columns other than 'sample_id' and 'chrom'
    represent population names.
    """
    return rf_q.drop(["sample_id", "chrom"], axis=1).columns.values


def _get_sample_names(rf_q: DataFrame):
    """
    Extract unique sample IDs from an RFMix Q-matrix DataFrame and
    convert to Arrow array.

    This function retrieves unique values from the 'sample_id' column
    of the input DataFrame and converts them to a PyArrow array.

    Parameters
    ----------
    rf_q (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
        Expected to have a 'sample_id' column.

    Returns
    -------
    pa.Array: A PyArrow array containing unique sample IDs.

    Example
    -------
    If rf_q has a 'sample_id' column with values ['sample1', 'sample2', 
    'sample1', 'sample3'], this function will return a PyArrow array
    containing ['sample1', 'sample2', 'sample3'].

    Note
    ----
    This function assumes that the 'sample_id' column exists in the
    input DataFrame. It uses PyArrow on GPU for efficient memory 
    management and interoperability with other data processing libraries.
    """
    if is_available() and isinstance(rf_q, DataFrame):
        return rf_q.sample_id.unique().to_arrow()
    else:
        return rf_q.sample_id.unique()


def _demo():
    from time import time
    df = DataFrame({
        'chromosome': ['chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1', 'chr1',
                       'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2', 'chr2'],
        'physical_position': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]
    })
    dask_matrix = from_array([
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ])
    sample_ids = ["data_1", "data_2", "data_3"]; pops = ["AFR", "EUR"]
    npops = len(pops)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    tic = time()
    bed_df = _generate_bed(df, dask_matrix, len(pops), col_names)
    toc = time()
    elapse = toc - tic
    return bed_df, elapse


def _old_intervals(data_matrix: Array, npops: int) -> List[int]:
    # Validate input dimensions
    if data_matrix.shape[1] % npops != 0:
        raise ValueError(f"Matrix column count ({data_matrix.shape[1]}) is not divisible by npops ({npops})")
    # Determine the number of columns to process
    num_cols = data_matrix.shape[1] // npops if npops == 2 else data_matrix.shape[1]
    all_indices = set()    
    for col in range(num_cols):
        # Extract column data
        col_data = data_matrix[:, col]
        # Compute differences between adjacent elements
        diffs = diff(col_data)
        # Fine indices where differences are non-zero (indicating a change)
        col_change_indices = where(diffs != 0)[0].compute()
        # Add these indices to the set of all change indices
        all_indices.update(col_change_indices)
    # Return a unique set of indices
    return sorted(all_indices)
