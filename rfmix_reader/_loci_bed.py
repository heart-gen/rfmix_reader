from tqdm import tqdm
from dask import config
from typing import List
import dask.dataframe as dd
from numpy import ndarray, full
from multiprocessing import cpu_count
from dask.array import (
    diff,
    Array,
    array,
    asarray,
    from_array,
    concatenate,
    expand_dims
)

try:
    import cupy as cp
    from cudf import DataFrame, concat
    config.set({"dataframe.backend": "cudf"})
    config.set({"array.backend": "cupy"})
    def is_available():
        return True
except ImportError:
    print("Warning: Using CPU!")
    import numpy as cp
    from pandas import DataFrame, concat
    config.set({"dataframe.backend": "pandas"})
    config.set({"array.backend": "numpy"})
    def is_available():
        return False

__all__ = [
    "generate_tagore_bed",
    "admix_to_bed_chromosome"
]

def generate_tagore_bed(
        loci: DataFrame, rf_q: DataFrame, admix: Array, sample_num: int,
        verbose: bool = True
) -> DataFrame:
    """
    Generate a BED (Browser Extensible Data) file formatted for TAGORE
    visualization.

    This function processes genomic data and creates a BED file suitable for
    visualization with TAGORE (https://github.com/jordanlab/tagore).

    Parameters:
        loci (DataFrame): A DataFrame containing genomic loci information.
        rf_q (DataFrame): A DataFrame containing recombination fraction
                          quantiles.
        admix (Array): An array of admixture proportions.
        sample_num (int): The sample number to process.
        verbose (bool, optional): If True, print progress information.
                                  Defaults to True.

    Returns:
        DataFrame: A DataFrame in BED format, annotated and ready for TAGORE
                   visualization.

    Note:
        This function relies on several helper functions:
        - admix_to_bed_chromosome: Converts admixture data to BED format for a
                                   specific individual.
        - _string_to_int: Converts specific columns in the BED DataFrame to
                          integer type (interal function).
        - _annotate_tagore: Adds annotation columns required for TAGORE
                            visualization (internal function).
    """
    # Convert admixture data to BED format for the specified sample
    bed = admix_to_bed_chromosome(loci, rf_q, admix, 0, verbose)
    # Get the name of the sample column (assumed to be the 4th column)
    sample_name = bed.columns[3]
    # Convert string columns to integer type
    bed = _string_to_int(bed, sample_name)
    # Annotate the BED file for TAGORE visualization
    return _annotate_tagore(bed, sample_name)


def _annotate_tagore(df: DataFrame, sample_name: str) -> DataFrame:
    """
    Annotate a DataFrame with additional columns for visualization purposes.

    This function expands the input DataFrame, adds annotation columns such as
    'feature', 'size', 'color', and 'chrCopy', and renames some columns for
    compatibility with visualization tools.

    Parameters:
        df (DataFrame): The input DataFrame to be annotated.
        sample_name (str): The name of the column containing sample data.

    Returns:
        DataFrame: The annotated DataFrame with additional columns.
    """
    # Define a color dictionary to map sample values to colors
    color_dict = {1:"#E64B35FF", 0:"#4DBBD5FF"}
    # Expand the DataFrame using the _expand_dataframe function
    expanded_df = _expand_dataframe(df, sample_name)
    # Initialize columns for feature and size
    expanded_df["feature"] = 0
    expanded_df["size"] = 1
    # Map the sample_name column to colors using the color_dict
    expanded_df["color"] = expanded_df[sample_name].map(color_dict)
    # Generate a repeating sequence of 1 and 2
    repeating_sequence = cp.tile(cp.array([1, 2]),
                                 int(cp.ceil(len(expanded_df) / 2)))[:len(expanded_df)]
    # Add the repeating sequence as a new column
    expanded_df['chrCopy'] = repeating_sequence
    # Drop the sample_name column and rename columns for compatibility
    return expanded_df.drop([sample_name], axis=1)\
                      .rename(columns={"chromosome": "#chr", "end": "stop"})


def _string_to_int(bed: DataFrame, sample_name: str) -> DataFrame:
    """
    Convert specific columns in a BED DataFrame from string to integer type.

    This function converts the sample_name column and the 'start' and 'end'
    columns to integer type. If CUDA is available, it also converts the
    DataFrame to a cuDF DataFrame for GPU-accelerated processing.

    Parameters:
        bed (DataFrame): The input BED DataFrame.
        sample_name (str): The name of the column containing sample data.

    Returns:
        DataFrame: The modified DataFrame with converted column types.
    """
    # Convert the columns to integer type
    bed[sample_name] = bed[sample_name].astype(int)
    bed["start"] = bed["start"].astype(int)
    bed["end"] = bed["end"].astype(int)
    # Check if CUDA is available
    if is_available():
        from cudf import from_pandas
        # Convert the pandas DataFrame to a cuDF DataFrame
        bed = from_pandas(bed)
    return bed


def _expand_dataframe(df: DataFrame, sample_name: str) -> DataFrame:
    """
    Expands a dataframe by duplicating rows based on a specified sample name
    column.

    For rows where the value in the sample name column is greater than 1, the
    function creates two sets of rows:
    1. The original rows with the sample name value decremented by 1.
    2. Rows with the sample name value set to either 1 or 0 based on the
       condition.

    The resulting dataframe is then sorted by 'chromosome', 'start', and the
    sample name column.

    Parameters:
        df (DataFrame): The input dataframe to be expanded.
        sample_name (str): The name of the column to be used for the expansion
                           condition.

    Returns:
        DataFrame: The expanded and sorted dataframe.
    """
    # Create a boolean mask for rows where sample_id > 1
    mask = df[sample_name] > 1
    # Create the first set of rows:
    # - For rows where mask is True: decrease sample_id by 1
    # - For rows where mask is False: keep original sample_id
    df1 = df.copy()
    df1.loc[mask, sample_name] -= 1
    # Create the second set of rows:
    # - For rows where mask is True: set sample_id to 1
    # - For rows where mask is False: set sample_id to 0
    df2 = df.copy()
    df2[sample_name] = cp.where(mask, 1, 0)
    # Concatenate the two dataframes vertically
    expanded_df = concat([df1, df2], ignore_index=True)
    # Sort the expanded DataFrame:
    # - First by 'chromosome' (ascending)
    # - Then by 'start' (ascending)
    # - Finally by sample_name (descending)
    # Reset the index after sorting
    return expanded_df.sort_values(by=['chromosome', 'start', sample_name],
                                   ascending=[True, True, False])\
                      .reset_index(drop=True)


def admix_to_bed_chromosome(
        loci: DataFrame, rf_q: DataFrame, admix: Array, sample_num: int,
        verbose: bool=True
) -> DataFrame:
    """
    Returns loci and admixture data to a BED (Browser Extensible Data) file for
    a specific chromosome.

    This function processes genetic loci data along with admixture proportions
    and returns BED format DataFrame for a specific chromosome.

    Parameters
    ----------
    loci : DataFrame
        A DataFrame containing genetic loci information. Expected to have
        columns for chromosome, position, and other relevant genetic markers.

    rf_q : DataFrame
        A DataFrame containing sample and population information. Used to derive
        sample IDs and population names.

    admix : Array
        A Dask Array containing admixture proportions. The shape should be
        compatible with the number of loci and populations.

    sample_num : int
       The column name including in data, will take the first population

    verbose : bool
       :const:`True` for progress information; :const:`False` otherwise.
       Default:`True`.

    Returns
    -------
    DataFrame: A DataFrame (pandas or cudf) in BED-like format with columns:
        'chromosome', 'start', 'end', and ancestry data columns.


    Notes
    -----
    - The function internally calls _generate_bed() to perform the actual BED
      formatting.
    - Column names in the output file are formatted as "{sample}_{population}".
    - The output file includes data for all chromosomes present in the input
      loci DataFrame.
    - Large datasets may require significant processing time and disk space.

    Example
    -------
    >>> loci, rf_q, admix = read_rfmix(prefix_path)
    >>> admix_to_bed_chromosome(loci_df, rf_q_df, admix_array, "chr22")
    """
    # Column annotations
    pops = _get_pops(rf_q)
    sample_ids = _get_sample_names(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    sample_name = f"{sample_ids[sample_num]}_{pops[0]}"
    # Generate BED dataframe
    ddf = _generate_bed(loci, admix, len(pops), col_names, sample_name, verbose)
    return ddf.compute()


def _generate_bed(
        df: DataFrame, dask_matrix: Array, npops: int,
        col_names: List[str], sample_name: str, verbose: bool
) -> DataFrame:
    """
    Generate BED records from loci and admixture data and subsets for specific
    chromosome.

    This function processes genetic loci data along with admixture proportions
    and returns the results for a specific chromosome.

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

    chrom : str
        The chromosome to generate BED format for.

    Returns
    -------
    DataFrame: A DataFrame (pandas or cudf) in BED-like format with columns:
        'chromosome', 'start', 'end', and ancestry data columns.

    Notes
    -----
    - The function internally calls _process_chromosome() to process each
      chromosome.
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
    if isinstance(dask_matrix, ndarray):
        dask_matrix = from_array(dask_matrix, chunks="auto")
    dask_df = dd.from_dask_array(dask_matrix, columns=col_names)
    ddf = dd.concat([ddf, dask_df], axis=1)
    del dask_df
    # Subset for chromosome
    results = []
    chromosomes = ddf["chromosome"].drop_duplicates().compute()
    for chrom in tqdm(sorted(chromosomes), desc="Processing Chromosomes",
                      disable=not verbose):
        chrom_group = ddf[ddf['chromosome'] == chrom]
        chrom_group = chrom_group.repartition(npartitions=parts)
        results.append(_process_chromosome(chrom_group, sample_name))
    return dd.concat(results, axis=0)


def _process_chromosome(group: dd.DataFrame, sample_name: str) -> DataFrame:
    """
    Process genetic data for a single chromosome to identify ancestry
    intervals.

    Converts genetic positions into BED-like intervals with constant ancestry,
    detecting change points where ancestry composition shifts.

    Parameters
    ----------
    group : dd.DataFrame
        Dask DataFrame containing genetic data for a single chromosome.
        Must contain columns: chromosome, physical_position, and [sample_name].
        Must be sorted by physical position.
    sample_name : str
        Name of the ancestry data column to preserve in output.

    Returns
    -------
    DataFrame
        BED-formatted DataFrame with columns:
        - chromosome (str/int): Chromosome identifier
        - start (int): Interval start position
        - end (int): Interval end position
        - [sample_name] (float): Ancestry proportion value

    Raises
    ------
    ValueError
        If input contains data for multiple chromosomes
        If physical positions are not sorted

    Notes
    -----
    Processing Workflow:
    1. Validates single-chromosome input
    2. Converts positions and ancestry data to Dask arrays
    3. Detects ancestry change points using _find_intervals
    4. Generates BED records for constant-ancestry intervals
    5. Returns formatted results as Dask DataFrame

    Example
    -------
    >>> group = dd.from_pandas(pd.DataFrame({
    ...     'chromosome': [1,1,1,1],
    ...     'physical_position': [100,200,300,400],
    ...     'pop1': [1,1,0,0]
    ... }), npartitions=1)
    >>> _process_chromosome(group, 'pop1').compute()
      chromosome  start  end  pop1
    0          1    100  200     1
    1          1    300  400     0
    """
    # Fetch chromosome
    chrom_val = group["chromosome"].drop_duplicates().compute()
    if len(chrom_val) != 1:
        raise ValueError(f"Only one chromosome expected got: {len(chrom_val)}")
    chrom_val = chrom_val.values[0]
    # Convert to a Dask array
    positions = group['physical_position'].to_dask_array(lengths=True)
    data_matrix = group[sample_name].to_dask_array(lengths=True)
    # Detect changes
    change_indices = _find_intervals(data_matrix)
    # Create BED records
    chrom_col, numeric_data = _create_bed_records(chrom_val, positions,
                                                  data_matrix, change_indices)
    cnames = ['chromosome', 'start', 'end'] + [sample_name]
    df_numeric = dd.from_dask_array(numeric_data, columns=cnames[1:])
    return df_numeric.assign(chromosome=chrom_val)[cnames]


def _find_intervals(data_matrix: Array) -> List[int]:
    """
    Detect ancestry change points in genetic data matrix.

    Identifies positions where ancestry composition changes across samples
    using column-wise differential analysis.

    Parameters
    ----------
    data_matrix : dask.array.Array
        2D array (positions × samples) of ancestry proportions
        dtype: numeric (float/int)

    Returns
    -------
    List[int]
        Sorted indices of ancestry change points (0-based)

    Notes
    -----
    Key Operations:
    1. Computes first-order differences across positions
    2. Identifies non-zero differentials (ancestry changes)
    3. Aggregates changes across all samples
    4. Returns sorted unique change points

    Performance Notes
    ----------------
    - Uses CuPy for GPU acceleration when available
    - Operates lazily until final compute() call
    - Memory efficient for large genomic datasets
    """
    # Vectorized diff operation across all relevant columns
    data_matrix = data_matrix.map_blocks(cp.asarray)
    diffs = diff(data_matrix, axis=0)
    # Find non-zero elements across all columns at once
    changes = diffs.map_blocks(lambda block: cp.where(block != 0)[0], dtype=int)
    # Compute chunks of indices
    change_indices = changes.compute()
    # Flatten and sort
    unique_changes = set(change_indices.tolist())
    return sorted(unique_changes)


def _create_bed_records(chrom_value, pos, data_matrix, idx):
    """
    Generate BED records from genetic intervals and ancestry data.

    Parameters
    ----------
    chrom_value : int or str
        Chromosome identifier for all records
    pos : dask.array.Array
        1D array of physical positions (int)
    data_matrix : dask.array.Array
        2D array of ancestry proportions (positions × samples)
    idx : Array
        Array of change point indices from _find_intervals

    Returns
    -------
    Tuple[Array, Array]
        (chromosome_col, numeric_data) where:
        - chromosome_col: Dask array of chromosome identifiers
        - numeric_data: Dask array with columns [start, end, sample_data]

    Notes
    -----
    Interval Construction Rules:
    - First interval starts at position 0
    - Subsequent intervals start at previous change point +1
    - Final interval ends at last physical position
    - Ancestry values taken from interval end points
    """
    idx = cp.asarray(idx)
    # Start and end index arrays
    start_indices = cp.concatenate([cp.array([0]), idx[:-1] + 1])
    end_indices = idx
    # Position slices
    start_col = pos[start_indices]; end_col = pos[end_indices]
    data_cols = data_matrix[end_indices]
    # Add last interval
    last_start = pos[int(end_indices[-1] + 1)]
    last_end = pos[int(end_indices[-1])]
    last_data = data_matrix[-1]
    start_col = concatenate([start_col, array([last_start])])
    end_col = concatenate([end_col, array([last_end])])
    data_cols = concatenate([data_cols, expand_dims(last_data, axis=0)])
    # Stack numeric columns: start, end, data_cols
    numeric_cols = cp.stack([cp.array(start_col.compute()),
                             cp.array(end_col.compute())] +
                            [cp.array(data_cols.compute())], axis=1)
    # Create string column using NumPy (CPU, safe for strings)
    chrom_col = from_array(full((numeric_cols.shape[0],), chrom_value)[:, None])
    # Convert numeric to Dask, then attach string column later
    numeric_data = from_array(numeric_cols)
    return chrom_col, numeric_data


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


def _load_admix():
    from rfmix_reader import read_rfmix
    basename = "/projects/b1213/resources/processed-data/local-ancestry"
    prefix_path = f"{basename}/rfmix-version/_m/"
    binary_dir = f"{basename}/rfmix-version/_m/binary_files/"
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def _viz_dev():
    loci, rf_q, admix = _load_admix()
    bed = admix_to_bed_chromosome(loci, rf_q, admix, 0)
    sample_name = bed.columns[3]
    bed = string_to_int(bed, sample_name)
    bed_df = annotate_tagore(bed, sample_name)
    return None
