"""
Documentation generation assisted by AI.
"""
import sys
from pathlib import Path
from zarr import Array as zArray
from psutil import virtual_memory
from typing import Tuple, List, Dict
from dask.array import Array, from_array
from dask.diagnostics import ProgressBar
from dask import config, delayed, compute
from dask.dataframe import from_dask_array
from dask.dataframe import from_pandas as dd_from_pandas

try:
    from torch.cuda import is_available, empty_cache
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    from dask_cudf import from_cudf
    from cudf import DataFrame, from_pandas, Series, concat, read_parquet
else:
    from gc import collect as empty_cache
    from dask.dataframe import from_pandas, concat
    from pandas import DataFrame, Series, read_parquet

__all__ = ["write_data", "write_imputed"]

def write_data(loci: DataFrame, rf_q: DataFrame, admix: Array,
               base_rows: int = 100_000, outdir: str = "./output",
               prefix: str = "local-ancestry", verbose: bool = False) -> None:
    """
    Write local ancestry data to a Parquet file per chromosome.

    This function combines loci information with local ancestry data and writes
    it to a Parquet files. The files includes chromosome, position, and
    ancestry haplotype information for each sample.

    Parameters:
    -----------
    loci : DataFrame (pandas or cuDF)
        A DataFrame containing loci information with at least the following
        columns:
        - 'chrom': Chromosome name or number.
        - 'pos': Position of the loci (1-based).
        Additional columns may include 'i' (used for filtering) or others.

    rf_q : DataFrame (pandas or cuDF)
        A DataFrame containing sample IDs and ancestry information. This is used
        to generate column names for the admixture data.

    admix : dask.Array
        A Dask array containing local ancestry haplotypes for each sample and
        locus.

    base_rows : int, optional (default=100,000)
        Controls how many rows each chunk (partition) of `admix` Dask array
        should have when rechunked. This is used to control memory consumption.
        Smaller chunks should be used to reduce memory.

    outdir : str, optional (default="./output")
        The directory where the output Parquet files will be saved.

    prefix : str, optional (default="local-ancestry")
        The file prefix for local ancestry data.

    verbose : bool, optional (default=False)
        Print out debugging information for partition shape and row matching.

    Returns:
    --------
    None
        Writes a Parquet files per chromosome to the specified output directory.

    Notes:
    ------
    - Updates columns names of loci if not already changed to ["chrom", "pos"].
    - The function handles both cuDF and pandas DataFrames. If cuDF is available,
      it converts `loci` to pandas before processing.

    Example:
    --------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir)
    >>> write_data(loci, rf_q, admix, outdir="./output", prefix="ancestry")
    # This will create ./output/ancestry.chr{1-22}.parquet files
    """
    from os import makedirs
    from pyarrow import Table
    from pyarrow.parquet import write_table
    # Memory optimization configuration
    config.set({"array.chunk-size": "256 MiB"})

    def write_partition(partition, path_template, loci_chunk, partition_idx):
        # Convert input partition to DataFrame
        output_path = str(path_template).format(partition_idx)
        df = DataFrame.from_records(partition, columns=col_names)
        df = concat([loci_chunk, df], axis=1)
        if is_available():
            # Leverage cuDF
            df.to_parquet(output_path, index=False)
        else:
            # Convert DataFrame to PyArrow Table
            write_table(Table.from_pandas(df), output_path)
        empty_cache()
        return None

    # Column name processing
    col_names = _get_names(rf_q)
    loci = _rename_loci_columns(loci)
    loci["pos"] = loci["pos"]
    loci["hap"] = loci['chrom'].astype(str) + '_' + loci['pos'].astype(str)
    loci = loci.drop(columns=["i"], errors="ignore")

    # Dynamic chunk sizing based on system memory
    target_rows = _get_target_rows(admix, base_rows)
    admix = admix.rechunk((target_rows, *admix.chunksize[1:]))

    # Split by chromosome
    chrom_data = _split_by_chrom(loci, admix)

    # Ensure output directory exists
    makedirs(outdir, exist_ok=True)

    # Processing each chromosome separately
    for chrom, (loci_chrom, admix_chrom) in chrom_data.items():
        out_template = Path(outdir) / f"{prefix}.{chrom}-{{}}.parquet"
        divisions = admix_chrom.numblocks[0] # Partition alignment

        if is_available():
            loci_ddf = from_cudf(loci_chrom, npartitions=divisions)
        else:
            loci_ddf = dd_from_pandas(loci_chrom, npartitions=divisions)

        # Rechunk to fix partition matching
        part_rows = loci_ddf.partitions[0].compute().shape[0]
        admix_chrom = admix_chrom.rechunk((part_rows, *admix_chrom.chunksize[1:]))
        _debug_partition_alignment(admix_chrom, loci_ddf, verbose=verbose)

        print(f"Processing chromosome {chrom} with {divisions} partitions...")
        # Parallel processing per chromosome
        tasks = [
            delayed(write_partition)(
                admix_chrom.blocks[i],
                out_template,
                loci_ddf.partitions[i],
                i
            ) for i in range(divisions)
        ]
        with ProgressBar():
            compute(*tasks) # Execute all tasks in parallel


def write_imputed(rf_q: DataFrame, admix: Array, variant_loci: DataFrame,
                  z: zArray, base_rows: int = 250_000, outdir: str = "./",
                  prefix: str = "ancestry-imputed", verbose: bool = False) -> None:
    """
    Process and write imputed local ancestry data to a Parquet file per
    chromosome.

    This function first cleans and aligns imputed local ancestry data with
    variant loci information, then writes the result to Parquet files per
    chromosome.

    Parameters:
    -----------
    rf_q : DataFrame (pandas or cuDF)
        A DataFrame containing sample IDs and ancestry information. This is used
        to generate column names for the admixture data.

    admix : dask.Array
        A Dask array containing local ancestry haplotypes for each sample and
        locus.

    variant_loci : DataFrame (pandas or cuDF)
        A DataFrame containing variant loci information. Must include columns for
        chromosome, position, and any merge-related columns used in data cleaning.

    z : zarr.Array
        An array used in the data cleaning process to align indices between
        admix and variant_loci.

    base_rows : int, optional (default=250,000)
        Controls how many rows each chunk (partition) of `admix` Dask array
        should have when rechunked. This is used to control memory consumption.
        Smaller chunks should be used to reduce memory.

    outdir : str, optional (default="./output")
        The directory where the output Parquet files will be saved.

    prefix : str, optional (default="ancestry-imputed")
        The file prefix for imputed local ancestry data.

    verbose : bool, optional (default=False)
        Print out debugging information for partition shape and row matching.

    Returns:
    --------
    None
        Writes an imputed local ancestry Parquet files per chromosome.

    Notes:
    ------
    - Calls `_clean_data_imp` to process and align the input data.
    - Uses `write_data` to output the cleaned data to Parquet files.
    - The resulting Parquet files includes chromosome, position, and ancestry
      probabilities for each sample at each locus.
    - Ensure that `variant_loci` contains necessary columns ('chrom' and 'pos').

    Example:
    --------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir)
    >>> loci.rename(columns={"chromosome": "chrom","physical_position": "pos"},
        inplace=True)
    >>> variant_loci = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
        how="outer", indicator=True).loc[:, ["chrom", "pos", "i", "_merge"]]
    >>> data_path = f"{basename}/local_ancestry_rfmix/_m"
    >>> z = interpolate_array(variant_loci, admix, data_output_dir)
    >>> write_imputed(rf_q, admix, variant_loci, z, outdir="./output",
        prefix="imputed-ancestry")
    # This will create ./output/imputed-ancestry.chr{1-22}.parquet files
    """
    loci_I, admix_I = _clean_data_imp(admix, variant_loci, z)
    write_data(loci_I, rf_q, admix_I, base_rows, outdir, prefix, verbose)


def _get_names(rf_q: DataFrame) -> List[str]:
    """
    Generate a list of sample names by combining sample IDs with N ancestries.

    This function creates a list of unique sample names by combining each unique
    sample ID with each ancestry. It handles both cuDF and pandas DataFrames.

    Parameters:
    -----------
    rf_q [DataFrame]: A DataFrame (pandas or cuDF) generated with `read_rfmix`.

    Returns:
    --------
    List[str]: A list of combined sample names in the format "sampleID_ancestry".

    Note:
    -----
    - The function assumes input from `read_rfmix`.
    - It uses cuDF-specific methods if available, otherwise falls back to pandas.
    """
    if is_available():
        sample_id = list(rf_q.sample_id.unique().to_pandas())
    else:
        sample_id = list(rf_q.sample_id.unique())
    ancestries = list(rf_q.drop(["sample_id", "chrom"], axis=1).columns.values)
    sample_names = [f"{sid}_{anc}" for anc in ancestries for sid in sample_id]
    return sample_names


def _rename_loci_columns(loci: DataFrame) -> DataFrame:
    """
    Rename columns in the loci DataFrame to standardized names.

    This function checks for the presence of 'chromosome' and 'physical_position'
    columns and renames them to 'chrom' and 'pos' respectively. If the columns
    are already named 'chrom' and 'pos', no changes are made.

    Parameters:
    -----------
    loci : DataFrame (pandas or cuDF)
        Input DataFrame containing loci information.

    Returns:
    --------
    DataFrame (pandas or cuDF)
        DataFrame with renamed columns.

    Notes:
    ------
    - If 'chromosome' is not present but 'chrom' is, no renaming occurs for that
      column.
    - If 'physical_position' is not present but 'pos' is, no renaming occurs for
      that column.
    - The function modifies the DataFrame in-place and also returns it.
    """
    rename_dict = {}
    if "chromosome" in loci.columns and "chrom" not in loci.columns:
        rename_dict["chromosome"] = "chrom"
    if "physical_position" in loci.columns and "pos" not in loci.columns:
        rename_dict["physical_position"] = "pos"
    if rename_dict:
        loci.rename(columns=rename_dict, inplace=True)
    return loci


def _get_target_rows(admix: Array, base_rows: int = 100_000):
    mem = virtual_memory().available
    row_size = admix.dtype.itemsize * admix.shape[1]
    return min(base_rows, int(mem * 0.5 / row_size)) # Use 50% of avail memory


def _split_by_chrom(
        loci: DataFrame, admix: Array) -> Dict[str, Tuple[DataFrame, Array]]:
    """
    Separates loci and admix by unique chromosome labels from 'chrom' column.

    Parameters:
    -----------
    loci : DataFrame (pandas or cuDF)
        DataFrame containing loci information with at least a 'chrom' column.

    admix : dask.Array
        A Dask array containing local ancestry haplotypes for each sample and
        locus.

    Returns:
    --------
    dict
        A dictionary where keys are chromosome names and values are tuples:
        { 'chromosome_label': (filtered_loci, filtered_admix) }
    """
    from collections import defaultdict
    chrom_dict = defaultdict(tuple)
    unique_chroms = loci["chrom"].unique().to_pandas() if is_available() else loci["chrom"].unique()
    # Create dictionary by chromosome
    for chrom in unique_chroms:
        # Extract loci for this chromosome
        loci_chrom = loci[loci["chrom"] == chrom]
        # Get row indices for this chromosome
        indices = loci_chrom.index.to_pandas() if is_available() else loci_chrom.index
        # Extract corresponding rows from admix
        admix_chrom = admix[indices.to_numpy()]
        # Store in dictionary
        chrom_dict[chrom] = (loci_chrom, admix_chrom)
    return dict(sorted(chrom_dict.items(), key=lambda item: item[1][0].shape[0]))


def _debug_partition_alignment(admix_arr, loci_ddf, verbose=False) -> None:
    """
    Debugging function to check partition alignment between admix_arr and
    loci_ddf. Exits with an error if partitions do not match.

    Parameters:
        admix_arr: A Dask array or similar object with `.blocks`.
        loci_ddf: A Dask DataFrame with `.partitions`.

    Returns:
        None. Exits the program if partitions do not align.
    """
    # Print rows of admix_arr partitions
    admix_rows = [block.shape[0] for block in admix_arr.blocks]
    if verbose:
        print("Admix array partition rows:", admix_rows)
    # Print rows of loci_ddf partitions
    loci_rows = [part.compute().shape[0] for part in loci_ddf.partitions]
    if verbose:
        print("Loci Dask DataFrame partition rows:", loci_rows)
    # Check if the number of partitions match
    if admix_arr.numblocks[0] != loci_ddf.npartitions:
        print("Error: Number of partitions do not match.")
        sys.exit(1)
    # Check if the rows of all corresponding partitions match
    for i, (admix_row, loci_row) in enumerate(zip(admix_rows, loci_rows)):
        if admix_row != loci_row:
            print(f"Error: Mismatch in partition {i}:",
                  f"Admix row = {admix_row}, Loci row = {loci_row}")
            sys.exit(1)
    if verbose:
        print("Number of partitions and rows match.")


def _clean_data_imp(admix: Array, variant_loci: DataFrame, z: zArray
                    ) -> Tuple[DataFrame, Array]:
    """
    Clean and align admixture data with variant loci information.

    This function processes admixture data and variant loci information,
    aligning them based on shared indices and filtering out unnecessary data.

    Parameters:
    -----------
    admix (Array): The admixture data array.
    variant_loci (DataFrame): A DataFrame containing variant and loci
                              information.
    z (zarr.Array): A zarr.Array object generated from `interpolate_array`.

    Returns:
    --------
    Tuple[DataFrame, Array]: A tuple containing:
        - loci_I (DataFrame): Cleaned and filtered variant and loci information
                              from imputed data.
        - admix_I (dask.Array): Cleaned and aligned admixture data from imputed
                                data.

    Note:
    -----
    - The function assumes the presence of an '_merge' column in variant_loci.
    - It uses dask arrays for efficient processing of large datasets.
    - The function handles both cuDF and pandas DataFrames, using cuDF if available.
    """
    daz = from_array(z, chunks=admix.chunksize)
    idx_arr = from_array(variant_loci[~(variant_loci["_merge"] ==
                                        "right_only")].index.to_numpy())
    admix_I = daz[idx_arr]
    mask = Series(False, index=variant_loci.index)
    mask.loc[idx_arr] = True
    if is_available():
        variant_loci = from_pandas(variant_loci)
    loci_I = variant_loci[mask].drop(["_merge"], axis=1)\
                               .reset_index(drop=True)
    return loci_I, admix_I
