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
    "admix_to_bed_individual"
]

def admix_to_bed_individual(
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
    >>> admix_to_bed_individual(loci_df, rf_q_df, admix_array, "chr22")
    """
    # Column annotations
    pops = get_pops(rf_q)
    sample_ids = get_sample_names(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    sample_name = f"{sample_ids[sample_num]}"

    # Generate BED dataframe
    ddf = _generate_bed(loci, admix, pops, col_names, sample_name, verbose)
    return ddf.compute()


def _generate_bed(
        df: DataFrame, dask_matrix: Array, pops: List[str],
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

    pops : List[str]
        A list of admixtured genetic populations

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
        results.append(_process_chromosome(chrom_group, sample_name, pops))

    return dd.concat(results, axis=0)


def _process_chromosome(
        group: dd.DataFrame, sample_name: str, pops: List[str]) -> DataFrame:
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
    change_indices = _find_intervals(data_matrix, len(pops))

    # Create BED records
    chrom_col, numeric_data = _create_bed_records(chrom_val, positions,
                                                  data_matrix, change_indices,
                                                  len(pops))
    cnames = ['chromosome', 'start', 'end'] + sample_cols
    df_numeric = dd.from_dask_array(numeric_data, columns=cnames[1:])
    return df_numeric.assign(chromosome=chrom_val)[cnames]


def _find_intervals(data_matrix: Array, npops: int) -> List[int]:
    """
    Detect ancestry change points in genetic data matrix.

    Identifies positions where ancestry composition changes across samples
    using column-wise differential analysis.

    Parameters
    ----------
    data_matrix : dask.array.Array
        2D array (positions × samples) of ancestry proportions
        dtype: numeric (float/int)
    npops : int
        The number of populations in the admixture data.

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
    if npops == 2:
        ancestry_vector = data_matrix[:, 0] # Select just one pop
    else:
        ancestry_vector = argmax(data_matrix, axis=1)

    ancestry_vector = ancestry_vector.map_blocks(cp.asarray, dtype=cp.int32)
    diffs = diff(ancestry_vector, axis=0)

    # Find non-zero elements across all columns at once
    changes = diffs.map_blocks(lambda block: cp.where(block != 0)[0], dtype=int)

    # Compute chunks of indices
    raw_indices = changes.compute()

    return sorted(set(raw_indices.tolist())) # Flatten and sort


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

    if npops == 2:
        ancestry_col = data_matrix[end_idx, 0]
        last_ancestry = data_matrix[max_idx, 0]
    else:
        ancestry_col = data_matrix[end_idx, :]
        last_ancestry = data_matrix[max_idx, :]

    # Add final interval
    last_start = pos[int(end_idx[-1] + 1)]
    last_end = pos[int(end_idx[-1])]

    start_col = concatenate([start_col, array([last_start])])
    end_col = concatenate([end_col, array([last_end])])
    ancestry_col = concatenate([ancestry_col,expand_dims(last_ancestry,axis=0)])

    # Stack numeric columns: start, end, ancestry_cols
    if npops == 2:
        numeric_cols = cp.stack([
            cp.array(start_col.compute()),
            cp.array(end_col.compute()),
            cp.array(ancestry_col.compute())
        ], axis=1)
    else:
        numeric_cols = cp.hstack([
            cp.array(start_col.compute().reshape(-1, 1)),
            cp.array(end_col.compute().reshape(-1, 1)),
            cp.array(ancestry_col.compute())
        ])

    chrom_col = from_array(full((numeric_cols.shape[0],), chrom_value)[:, None])
    numeric_data = from_array(numeric_cols)
    return chrom_col, numeric_data


def _load_real_data():
    from rfmix_reader import read_rfmix
    basename = "/projects/b1213/resources/processed-data/local-ancestry"
    prefix_path = f"{basename}/rfmix-version/_m/"
    binary_dir = f"{basename}/rfmix-version/_m/binary_files/"
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def _load_simu_data(pop=2):
    from pathlib import Path
    from rfmix_reader import read_rfmix
    basename = "/projects/p32505/projects/rfmix_reader-benchmarking/input/simulations"
    pop_loc = "two_populations" if pop == 2 else "three_populations"
    prefix_path = Path(basename) / pop_loc / "_m/rfmix-out/"
    binary_dir = prefix_path / "binary_files"
    if binary_dir.exists():
        return read_rfmix(prefix_path, binary_dir=binary_dir)
    else:
        return read_rfmix(prefix_path, binary_dir=binary_dir,
                          generate_binary=True)


def _testing_simulation(pop_num, sample_num):
    loci, rf_q, admix = _load_simu_data(pop_num)
    return admix_to_bed_individual(loci, rf_q, admix, sample_num)


def _testing_real(sample_num):
    loci, rf_q, admix = _load_real_data()
    return admix_to_bed_individual(loci, rf_q, admix, sample_num)
