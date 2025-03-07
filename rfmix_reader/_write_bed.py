"""
Documentation generation assisted by AI.
"""
from typing import Tuple, List
from zarr import Array as zArray
from dask.array import Array, from_array
from dask.diagnostics import ProgressBar
from dask.dataframe import from_dask_array, concat
from dask.dataframe import from_pandas as dd_from_pandas

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, from_pandas, Series
else:
    from pandas import DataFrame, Series
    from dask.dataframe import from_pandas

__all__ = ["write_bed", "write_imputed"]

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
    def process_partition(df, ii):
        """Writes each partition to CSV without loading all data into memory."""
        df.to_csv(f"{outdir}/{outfile}", sep="\t", mode="a", index=False,
                  header=(ii == 0))
        return df  # Return df to satisfy Dask's expected return format
    # Convert Dask Array to Dask DataFrame
    admix_ddf = from_dask_array(admix, columns=_get_names(rf_q))
    # Compute optimal number of partitions (targeting ~500k rows per partition)
    npartitions = max(10, min(50, admix.shape[0] // 500_000))
    # Re-partition data into chunks
    admix_ddf = admix_ddf.repartition(npartitions=npartitions)
    # Fix loci column names
    loci = _rename_loci_columns(loci)
    loci["end"] = loci["pos"] + 1
    loci["hap"] = loci['chrom'].astype(str)+'_'+loci['pos'].astype(str)
    # Ensure loci is a Pandas DataFrame
    if is_available():
        loci = loci.to_pandas()
    else:
        loci.drop(["i"], axis=1, inplace=True)
    # Convert Pandas DataFrame to Dask DataFrame
    loci_ddf = dd_from_pandas(loci, npartitions=admix_ddf.npartitions)
    loci_ddf = loci_ddf.repartition(npartitions=admix_ddf.npartitions)
    # Concatenate loci and admix DataFrames
    bed_ddf = concat([loci_ddf, admix_ddf], axis=1)
    # Apply function to all partitions using map_partitions
    print(f"Processing {npartitions} partitions...")
    with ProgressBar():
        _ = bed_ddf.to_delayed()
        _ = [process_partition(df.compute(), ii) for ii, df in enumerate(_)]


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
