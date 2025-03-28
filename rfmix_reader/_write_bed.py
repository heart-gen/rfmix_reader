"""
Documentation generation assisted by AI.
"""
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List
from zarr import Array as zArray
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
    from cudf import DataFrame, from_pandas, Series, concat
else:
    from pandas import DataFrame, Series
    from gc import collect as empty_cache
    from dask.dataframe import from_pandas, concat

__all__ = ["write_bed", "write_imputed"]


def write_bed(loci: DataFrame, rf_q: DataFrame, admix: Array,
              outdir: str = "./", outfile: str = "local-ancestry.bed") -> None:
    """
    Write local ancestry data to a BED-format file.

    This function combines loci information with local ancestry data and writes
    it to a BED-format file. The BED file includes chromosome, position, and
    ancestry haplotype information for each sample.

    Parameters:
    -----------
    loci : DataFrame (pandas or cuDF)
        A DataFrame containing loci information with at least the following columns:
        - 'chrom': Chromosome name or number.
        - 'pos': Position of the loci (1-based).
        Additional columns may include 'i' (used for filtering) or others.

    rf_q : DataFrame (pandas or cuDF)
        A DataFrame containing sample IDs and ancestry information. This is used to
        generate column names for the admixture data.

    admix : dask.Array
        A Dask array containing local ancestry haplotypes for each sample and locus.

    outdir : str, optional (default="./")
        The directory where the output BED file will be saved.

    outfile : str, optional (default="local-ancestry.bed")
        The name of the output BED file.

    Returns:
    --------
    None
        Writes a BED-format file to the specified output directory.

    Notes:
    ------
    - Updates columns names of loci if not already changed to ["chrom", "pos"].
    - The function adds an 'end' column (BED format requires a start and end
      position).
    - The function handles both cuDF and pandas DataFrames. If cuDF is available,
      it converts `loci` to pandas before processing.
    - The BED file is written in chunks to handle large datasets efficiently.
      The header is written only for the first chunk.

    Example:
    --------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir)
    >>> write_bed(loci, rf_q, admix, outdir="./output", outfile="ancestry.bed")
    # This will create ./output/ancestry.bed with the processed data

    Example Output Format:
    ----------------------
    chrom   pos   end   hap   sample1_ancestry1   sample2_ancestry1 ...
    chr1    1000  1001  chr1_1000   0   1 ...
    """
    from numpy import int32
    # Memory optimization configuration
    config.set({"array.chunk-size": "158 MiB"})
    target_rows = 100_000  # Reduced partition size for memory efficiency
    def write_partition(partition, path, loci_chunk, first=False):
        """GPU-aware partitioned writing with reduced memory footprint"""
        path = Path(path)
        if is_available():
            df = DataFrame.from_records(partition, columns=col_names)
            df = concat([loci_chunk, df], axis=1)
            df.to_pandas().to_csv(path, sep="\t", mode="a",
                                header=first and not path.exists(), index=False)
        else:
            df = loci_chunk.join(DataFrame(partition, columns=col_names))
            df.to_csv(path, sep="\t", mode="a",
                      header=first and not path.exists(), index=False)
        return None
    # Column name processing
    col_names = _get_names(rf_q)
    loci = _rename_loci_columns(loci)
    loci["pos"] = loci["pos"].astype(int32)
    loci["end"] = (loci["pos"] + 1).astype(int32)
    loci["hap"] = loci['chrom'].astype(str) + '_' + loci['pos'].astype(str)
    loci = loci.drop(columns=["i"], errors="ignore")
    # Memory-conscious rechunking
    admix = admix.rechunk((target_rows, *admix.chunksize[1:]))
    # Align partitions between loci and admix data
    divisions = admix.numblocks[0]
    if is_available():
        loci_ddf = from_cudf(loci, npartitions=divisions)
    else:
        loci_ddf = dd_from_pandas(loci, npartitions=divisions)
    # Stream processing pipeline
    output_path = f"{outdir}/{outfile}"
    print(f"Processing {divisions} partitions...")
    tasks = [
        delayed(write_partition)(
            admix.blocks[i].compute(),
            output_path,
            loci_ddf.partitions[i].compute(),
            first=(i == 0)) for i in range(divisions)
    ]
    for i, task in enumerate(tqdm(tasks, desc="Processing tasks",
                                  total=len(tasks))):
        compute(task)
        torch.cuda.empty_cache()
    # mini_batch = 5
    # total_batches = (len(tasks) + mini_batch - 1) // mini_batch
    # with tqdm(total=total_batches, desc="Processing tasks") as pbar:
    #     for i in range(0, len(tasks), mini_batch):
    #         batch = tasks[i : i + mini_batch]
    #         compute(*batch)
    #         empty_cache()
    #         pbar.update(1)


def write_bed(loci: DataFrame, rf_q: DataFrame, admix: Array, outdir: str="./",
              outfile: str = "local-ancestry.bed") -> None:
    """
    Write local ancestry data to a BED-format file.

    This function combines loci information with local ancestry data and writes
    it to a BED-format file. The BED file includes chromosome, position, and
    ancestry haplotype information for each sample.

    Parameters:
    -----------
    loci : DataFrame (pandas or cuDF)
        A DataFrame containing loci information with at least the following columns:
        - 'chrom': Chromosome name or number.
        - 'pos': Position of the loci (1-based).
        Additional columns may include 'i' (used for filtering) or others.

    rf_q : DataFrame (pandas or cuDF)
        A DataFrame containing sample IDs and ancestry information. This is used to
        generate column names for the admixture data.

    admix : dask.Array
        A Dask array containing local ancestry haplotypes for each sample and locus.

    outdir : str, optional (default="./")
        The directory where the output BED file will be saved.

    outfile : str, optional (default="local-ancestry.bed")
        The name of the output BED file.

    Returns:
    --------
    None
        Writes a BED-format file to the specified output directory.

    Notes:
    ------
    - Updates columns names of loci if not already changed to ["chrom", "pos"].
    - The function adds an 'end' column (BED format requires a start and end
      position).
    - The function handles both cuDF and pandas DataFrames. If cuDF is available,
      it converts `loci` to pandas before processing.
    - The BED file is written in chunks to handle large datasets efficiently.
      The header is written only for the first chunk.

    Example:
    --------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir)
    >>> write_bed(loci, rf_q, admix, outdir="./output", outfile="ancestry.bed")
    # This will create ./output/ancestry.bed with the processed data

    Example Output Format:
    ----------------------
    chrom   pos   end   hap   sample1_ancestry1   sample2_ancestry1 ...
    chr1    1000  1001  chr1_1000   0   1 ...
    """
    def process_partition(df, file_path, first):
        """Writes each partition efficiently."""
        write_header = first and (not file_path.exists())
        if is_available():
            df.to_pandas().to_csv(file_path, sep="\t", mode="a", index=False,
                                  header=write_header)
        else:
            df.to_csv(file_path, sep="\t", mode="a", index=False,
                      header=write_header)
        return None
    # Convert Dask Array to Dask DataFrame
    col_names = _get_names(rf_q)
    admix_ddf = from_dask_array(admix, columns=col_names)
    # Optimal number of partitions (targeting ~500k rows per partition)
    target_rows_per_partition = 500_000
    npartitions = max(10, min(50, admix.shape[0] // target_rows_per_partition))
    ##admix_ddf = admix_ddf.repartition(npartitions=npartitions)
    # Fix loci column names
    loci = _rename_loci_columns(loci)
    loci["end"] = loci["pos"] + 1
    loci["hap"] = loci['chrom'].astype(str)+'_'+loci['pos'].astype(str)
    # Keep loci as Dask DataFrame
    if is_available():
        loci_ddf = from_cudf(loci, npartitions=npartitions)
        admix_ddf = admix_ddf.map_partitions(lambda df: DataFrame(df))
    else:
        loci.drop(columns=["i"], errors="ignore", inplace=True)
        loci_ddf = dd_from_pandas(loci, npartitions=npartitions)
    ##loci_ddf = loci_ddf.repartition(npartitions=admix_ddf.npartitions)
    # Concatenate loci and admix DataFrames
    bed_ddf = concat([loci_ddf, admix_ddf], axis=1)
    bed_ddf = bed_ddf.repartition(npartitions=npartitions)
    meta = DataFrame(columns=bed_ddf.columns)
    if is_available():
        meta = meta.to_pandas()
    # Write partitions efficiently
    print(f"Processing {npartitions} partitions...")
    file_path = Path(f"{outdir}/{outfile}")
    with ProgressBar():
        bed_ddf.map_partitions(process_partition, file_path, first=True,
                               meta=meta).compute()


def write_imputed(rf_q: DataFrame, admix: Array, variant_loci: DataFrame,
                  z: zArray, outdir: str = "./",
                  outfile: str = "local-ancestry.imputed.bed") -> None:
    """
    Process and write imputed local ancestry data to a BED-format file.

    This function cleans and aligns imputed local ancestry data with variant loci
    information, then writes the result to a BED-format file.

    Parameters:
    -----------
    rf_q : DataFrame (pandas or cuDF)
        A DataFrame containing sample IDs and ancestry information. Used to
        generate column names for the admixture data.

    admix : dask.Array
        A Dask array containing local ancestry probabilities for each sample and locus.

    variant_loci : DataFrame (pandas or cuDF)
        A DataFrame containing variant loci information. Must include columns for
        chromosome, position, and any merge-related columns used in data cleaning.

    z : zarr.Array
        An array used in the data cleaning process to align indices between
        admix and variant_loci.

    outdir : str, optional (default="./")
        The directory where the output BED file will be saved.

    outfile : str, optional (default="local-ancestry.imputed.bed")
        The name of the output BED file.

    Returns:
    --------
    None
        Writes an imputed local ancestry BED-format file to the specified location.

    Notes:
    ------
    - Calls `_clean_data_imp` to process and align the input data.
    - Uses `write_bed` to output the cleaned data in BED format.
    - The resulting BED file includes chromosome, position, and ancestry
      probabilities for each sample at each locus.
    - Ensure that `variant_loci` contains necessary columns for BED format
      (typically 'chrom' and 'pos' or equivalent).

    Example:
    --------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir)
    >>> loci.rename(columns={"chromosome": "chrom","physical_position": "pos"}, inplace=True)
    >>> variant_loci = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"], how="outer", indicator=True).loc[:, ["chrom", "pos", "i", "_merge"]]
    >>> data_path = f"{basename}/local_ancestry_rfmix/_m"
    >>> z = interpolate_array(variant_loci, admix, data_output_dir)
    >>> write_imputed(rf_q, admix, variant_loci, z, outdir="./output", outfile="imputed_ancestry.bed")
    # This will create ./output/imputed_ancestry.bed with the processed data
    """
    loci_I, admix_I = _clean_data_imp(admix, variant_loci, z)
    write_bed(loci_I, rf_q, admix_I, outdir, outfile)


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
    loci_I = variant_loci[mask].drop(["i", "_merge"], axis=1)\
                               .reset_index(drop=True)
    return loci_I, admix_I
