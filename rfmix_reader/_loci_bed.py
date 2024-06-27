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

__all__ = ["export_loci_admix_to_bed"]

def export_loci_admix_to_bed(
        loci: DataFrame, rf_q: DataFrame, admix: Array,
        output_dir: str = "bed_results", verbose: bool = True) -> None:
    """
    Export loci and admixture data to a BED (Browser Extensible Data) file.

    This function processes genetic loci data along with admixture proportions and 
    writes the results to a BED format file. The output file includes sample-specific 
    admixture proportions for each population.

    Parameters
    ----------
    loci : DataFrame
        A DataFrame containing genetic loci information. Expected to have columns 
        for chromosome, position, and other relevant genetic markers.

    rf_q : DataFrame
        A DataFrame containing sample and population information. Used to derive 
        sample IDs and population names.

    admix : Array
        A Dask Array containing admixture proportions. The shape should be 
        compatible with the number of loci and populations.

    output_dir : str (optional)
        The path for the output BED file. Default: bed_results

    Returns
    -------
    None
        The function writes data to a file and doesn't return any value.


    Notes
    -----
    - The function internally calls _generate_bed() to perform the actual file writing.
    - Column names in the output file are formatted as "{sample}_{population}".
    - The output file includes data for all chromosomes present in the input 
      loci DataFrame.
    - Large datasets may require significant processing time and disk space.

    Example
    -------
    >>> loci, rf_q, admix = read_rfmix(prefix_path)
    >>> export_loci_admix_to_bed(loci_df, rf_q_df, admix_array)
    """
    # Column annotations
    pops = _get_pops(rf_q)
    sample_ids = _get_sample_names(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    # Generate BED dataframe
    _generate_bed(loci, admix, len(pops), col_names)
    return None


def _generate_bed(
        df: DataFrame, dask_matrix: Array, npops: int,
        col_names: List[str], output_dir: str, verbose: bool
) -> None:
    """
    Generate BED records from loci and admixture data and write them to Parquet files.

    This function processes genetic loci data along with admixture proportions and 
    writes the results to Parquet files, one for each chromosome. The output files 
    include sample-specific admixture proportions for each population.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing genetic loci information from `read_rfmix`

    dask_matrix : Array
        A Dask Array containing admixture proportions. The shape should be 
        compatible with the number of loci and populations. This is from
        `read_rfmix`.

    npops : int
        The number of populations in the admixture data.

    col_names : List[str]
        A list of column names for the admixture data. These should be formatted 
        as "{sample}_{population}".

    output_dir : str
        The directory where the output Parquet files will be written. Each 
        chromosome will have its own Parquet file.

    verbose : bool
        If True, progress information will be printed to the console.

    Returns
    -------
    None
        The function writes data to Parquet files and doesn't return any value.

    Side Effects
    ------------
    - Creates or overwrites Parquet files in the specified output directory.
    - Prints progress information to the console if verbose mode is enabled.

    Notes
    -----
    - The function internally calls _process_chromosome() to process each chromosome.
    - The output files are named based on the chromosome they contain data for.
    - Large datasets may require significant processing time and disk space.
    """
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
    for chrom in tqdm(sorted(chromosomes), desc="Processing Chromosomes",
                      disable=not verbose):
        chrom_group = ddf[ddf['chromosome'] == chrom]
        bed_records = _process_chromosome(chrom_group, npops, col_names)
        # Write results to Parquet file
        out_path = join(output_dir, chrom)
        bed_records.to_parquet(out_path, write_index=False)
        # Clear memory
        del bed_records, chrom_group
    if verbose:
        print(f"Results written to {out_dir}")
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


def _create_bed_records(chrom_value, pos, data_matrix, idx):
    """
    Create BED records for a given chromosome.

    This function generates BED (Browser Extensible Data) records for a specific chromosome
    by processing the positions and admixture data. It returns a concatenated array containing
    chromosome, start position, end position, and admixture data for each interval.

    Parameters
    ----------
    chrom_value : int or str
        The chromosome identifier (e.g., chromosome number or name).

    pos : dask.array
        A dask array containing the positions of genetic loci.

    data_matrix : dask.array
        A dask array containing admixture proportions. The shape should be compatible
        with the number of loci and populations.

    idx : dask.array
        A dask array of indices indicating the change points in the positions array.

    Returns
    -------
    dask.array
        A concatenated dask array with columns for chromosome, start position, end position,
        and admixture data for each interval. The shape of the array is (n_intervals, n_columns),
        where n_columns = 3 + n_pops (chromosome, start, end, and admixture proportions).

    Notes
    -----
    - The function assumes that the input arrays are properly aligned and that the indices
      in `idx` are valid for the `pos` and `data_matrix` arrays.
    - The last interval is handled separately to ensure all data is included.
    """
    # Create start indices
    start_indices = concatenate([array([0]), idx[:-1] + 1])
    # Create arrays for each column
    chrom_col = full(idx.shape, chrom_value)
    start_col = pos[start_indices]
    end_col = pos[idx]
    data_cols = data_matrix[idx]
    # Add the last interval
    last_start = pos[idx[-1] + 1]
    last_end = pos[-1]
    last_data = data_matrix[-1]
    # Concatenate all data
    chrom_col = concatenate([chrom_col, array([chrom_value])])
    start_col = concatenate([start_col, array([last_start])])
    end_col = concatenate([end_col, array([last_end])])
    data_cols = concatenate([data_cols, last_data.reshape(1, -1)])
    # Combine all columns
    return concatenate([chrom_col[:, None], start_col[:, None],
                        end_col[:, None], data_cols], axis=1)


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


def _testing():
    from rfmix_reader import read_rfmix, create_binaries
    # prefix_path = "../examples/two_populations/out/"
    prefix_path = "/dcs05/lieber/hanlab/jbenjami/projects/"+\
        "localQTL_manuscript/local_ancestry_rfmix/_m/"
    #create_binaries(prefix_path)
    loci, rf_q, admix = read_rfmix(prefix_path)
    export_loci_admix_to_bed(loci, rf_q, admix)
    return None
