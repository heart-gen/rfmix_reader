from tqdm import tqdm
from dask import config
import dask.dataframe as dd
from numpy import ndarray, full
from typing import List, Union, Tuple
from multiprocessing import cpu_count
from dask.array import (
    diff,
    Array,
    array,
    argmax,
    from_array,
    concatenate,
    expand_dims
)

from ._utils import get_pops, get_sample_names

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    import cupy as cp
    from cudf import DataFrame
    config.set({"dataframe.backend": "cudf"})
    config.set({"array.backend": "cupy"})
else:
    import numpy as cp
    from pandas import DataFrame
    config.set({"dataframe.backend": "pandas"})
    config.set({"array.backend": "numpy"})

__all__ = [
    "admix_to_bed_individual"
]

def admix_to_bed_individual(
        loci: DataFrame, rf_q: DataFrame, admix: Array, sample_num: int,
        chunk_size: int = 10_000, min_segment: int = 3, verbose: bool=True
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

    chunk_size : int, optional
        Size of chunks to process at once (default=10_000)
        Adjust based on available memory

    min_segment : int, optional
        Minimum length of a segment to consider it a true change (default=3)

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
    >>> admix_to_bed_individual(loci_df, rf_q_df, admix_array, "chr22")
    """
    # Column annotations
    pops = get_pops(rf_q)
    sample_ids = get_sample_names(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    sample_name = f"{sample_ids[sample_num]}"

    # Generate BED dataframe
    ddf = _generate_bed(loci, admix, pops, col_names, sample_name, verbose,
                        chunk_size, min_segment)
    return ddf.compute()


def _generate_bed(
        df: DataFrame, dask_matrix: Array, pops: List[str],
        col_names: List[str], sample_name: str, verbose: bool,
        chunk_size: int, min_segment: int
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

    pops : List[str]
        A list of admixtured genetic populations

    col_names : List[str]
        A list of column names for the admixture data. These should be formatted
        as "{sample}_{population}".

    chrom : str
        The chromosome to generate BED format for.

    chunk_size : int, optional
        Size of chunks to process at once. Adjust based on available memory.

    min_segment : int
        Minimum length of a segment to consider for a true change.

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
    for chrom in tqdm(sorted(chromosomes), desc="Processing chromosomes",
                      disable=not verbose):
        chrom_group = ddf[ddf['chromosome'] == chrom]
        chrom_group = chrom_group.repartition(npartitions=parts)
        results.append(_process_chromosome(chrom_group, sample_name, pops,
                                           chunk_size, min_segment))

    return dd.concat(results, axis=0)


def _process_chromosome(
        group: dd.DataFrame, sample_name: str, pops: List[str],
        chunk_size: int, min_segment: int
) -> DataFrame:
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

    pops : List[str]
        A list of admixtured populations

    chunk_size : int, optional
        Size of chunks to process at once. Adjust based on available memory.

    min_segment : int, optional
        Minimum length of a segment to consider it a true change.

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
    target_samples = [f"{sample_name}_{pop}" for pop in pops]
    sample_cols = [col for col in group.columns if col in target_samples]
    data_matrix = group[sample_cols].to_dask_array(lengths=True)

    # Detect changes
    change_indices = _find_intervals(data_matrix, chunk_size, min_segment)

    # Create BED records
    chrom_col, numeric_data = _create_bed_records(chrom_val, positions,
                                                  data_matrix, change_indices,
                                                  len(pops))
    cnames = ['chromosome', 'start', 'end'] + sample_cols
    df_numeric = dd.from_dask_array(numeric_data, columns=cnames[1:])
    return df_numeric.assign(chromosome=chrom_val)[cnames]


def _find_intervals(data_matrix: Array, chunk_size: int,
                    min_segment_length: int) -> List[int]:
    """
    Detect ancestry change points in genetic data matrix.

    Parameters
    ----------
    data_matrix : dask.array.Array
        2D array (positions × populations) of ancestry counts
        Each row sums to 2 (diploid), with values 0,1,2 per ancestry
        Shape example: (175_000, 2) for two populations

    chunk_size : int, optional
        Size of chunks to process at once. Adjust based on available memory.

    min_segment_length : int, optional
        Minimum length of a segment to consider it a true change.

    Returns
    -------
    List[int]
        Sorted indices of ancestry change points (0-based)
    """
    # Get dimensions
    n_positions = data_matrix.shape[0]
    n_chunks = (n_positions + chunk_size - 1) // chunk_size
    # Initialize change points list
    change_points = set([0])  # Always include start point
    # Process data in chunks with overlap to handle boundaries
    overlap = min_segment_length + 1
    last_state = None
    for chunk_idx in range(n_chunks):
        # Calculate chunk boundaries
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size + overlap, n_positions)
        # Get chunk data
        chunk = data_matrix[start_idx:end_idx].compute()
        # Find state changes within chunk
        for pos in range(1, len(chunk)):
            # Compare full state vectors
            current_state = tuple(chunk[pos])
            prev_state = tuple(chunk[pos-1])
            if current_state != prev_state:
                # Check if change persists for min_segment_length
                is_stable_change = True
                if pos + min_segment_length <= len(chunk):
                    future_states = chunk[pos:pos + min_segment_length]
                    # Check if the new state is stable
                    if not all(cp.array_equal(future_states[i], chunk[pos])
                             for i in range(min_segment_length)):
                        is_stable_change = False
                else:
                    # If we're near chunk boundary,
                    # mark for checking in next chunk
                    is_stable_change = False
                if is_stable_change:
                    global_pos = start_idx + pos
                    change_points.add(global_pos)
        # Save last state for next chunk comparison
        last_state = tuple(chunk[-1])
    # Add final position
    change_points.add(n_positions - 1)
    return sorted(change_points)


def _create_bed_records(
        chrom_value: Union[int, str], pos: Array, data_matrix: Array,
        idx: List[int], npops: int) -> Tuple[Array, Array]:
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
    idx : List[int]
        List of change point indices from _find_intervals
    npops : int
        The number of populations in the admixture data.

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

    if len(idx) == 0:
        ancestry = data_matrix[-1, 0] if npops == 2 else data_matrix[-1, :]
        start_col = pos[0]; end_col = pos[-1]
        chrom_col = from_array(full((1,), chrom_value)[:, None])
        ancestry_col = ancestry.compute().reshape(1, -1) if npops > 2 else [ancestry.compute()]
        numeric_cols = cp.hstack([
            cp.array([start_col.compute()]).reshape(-1, 1),
            cp.array([end_col.compute()]).reshape(-1, 1),
            cp.array(ancestry_col).reshape(1, -1)
        ])
        return chrom_col, from_array(numeric_cols)

    # Check if last interval has room for extension
    max_idx = len(pos) - 1
    final_idx = int(idx[-1])
    next_idx = final_idx + 1 if final_idx + 1 <= max_idx else final_idx

    # Start and end index arrays
    start_idx = cp.concatenate([cp.array([0]), idx[:-1] + 1])
    end_idx = idx

    # Position slices
    start_col = pos[start_idx]; end_col = pos[end_idx]

    ancestry_col = data_matrix[end_idx, :]
    last_ancestry = data_matrix[max_idx, :]

    # Add final interval
    last_end_index = int(end_idx[-1])
    if last_end_index + 1 < len(pos):
        last_start = pos[last_end_index + 1]
    else:
        last_start = pos[last_end_index]
    last_end = pos[int(end_idx[-1])]

    start_col = concatenate([start_col, array([last_start])])
    end_col = concatenate([end_col, array([last_end])])
    ancestry_col = concatenate([ancestry_col,expand_dims(last_ancestry,axis=0)])

    # Stack numeric columns: start, end, ancestry_cols
    numeric_cols = cp.hstack([
        cp.array(start_col.compute().reshape(-1, 1)),
        cp.array(end_col.compute().reshape(-1, 1)),
        cp.array(ancestry_col.compute())
    ])

    chrom_col = from_array(full((numeric_cols.shape[0],), chrom_value)[:, None])
    numeric_data = from_array(numeric_cols)
    return chrom_col, numeric_data
