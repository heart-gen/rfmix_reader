import seaborn as sns
from dask import config
from dask.array import Array
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Union, List, Optional

from ._utils import get_pops
from ._loci_bed import admix_to_bed_individual

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    import cupy as cp
    from cudf import DataFrame, concat
    config.set({"dataframe.backend": "cudf"})
    config.set({"array.backend": "cupy"})
else:
    import numpy as cp
    from pandas import DataFrame, concat
    config.set({"dataframe.backend": "pandas"})
    config.set({"array.backend": "numpy"})

__all__ = [
    "save_multi_format",
    "generate_tagore_bed",
    "plot_global_ancestry",
    "plot_ancestry_by_chromosome",
]

def plot_global_ancestry(
        rf_q: DataFrame, title: str = "Global Ancestry Proportions",
        palette: Union[str,List[str]] = 'tab10', figsize: Tuple[int,int] = (16,6),
        save_path: Optional[str] = "global_ancestry",
        show_labels: bool = False, sort_by: Optional[str] = None, **kwargs
) -> None:
    """
    Plot global ancestry proportions across all individuals.

    Parameters:
    -----------
    rf_q : DataFrame

    title : str, optional
        Plot title (default: "Global Ancestry Proportions")

    palette : Union[str, List[str]], optional
        Colormap name (matplotlib colormap) or list of color codes (default: 'tab10')

    figsize : Tuple[int, int], optional
        Figure dimensions in inches (width, height) (default: (16, 6))

    save_path : Optional[str], optional
        Base filename for saving plots (without extension). If None, shows interactive plot.
        (default: "global_ancestry")

    show_labels : bool, optional
        Display individual IDs on x-axis (default: False)

    sort_by : Optional[str], optional
        Ancestry column name to sort individuals by (default: None)

    **kwargs : dict
        Additional arguments passed to save_multi_format()

    Example:
    -------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir=binary_dir)
    >>> plot_global_ancestry(rf_q, dpi=300, bbox_inches="tight")
    """
    from pandas import Series
    from numpy import arange

    ancestry_df = _get_global_ancestry(rf_q)
    if hasattr(ancestry_df, "to_pandas"):
        ancestry_df = ancestry_df.to_pandas()

    if sort_by and sort_by in ancestry_df.columns:
        ancestry_df = ancestry_df.sort_values(by=sort_by, ascending=False)

    plt.close('all')  # Clear previous figures
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # Dynamic width adjustment
    bar_width = 0.8 if len(ancestry_df) < 500 else 1.0

    # Universe color extractor method
    npop = len(ancestry_df.columns)
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        colors = cmap.colors[:npop]
    else:
        colors = palette # Use provided list

    # Stacked bars with optimized drawing
    bottom = Series([0] * len(ancestry_df), index=ancestry_df.index)
    x = arange(len(ancestry_df))

    for i, ancestry in enumerate(ancestry_df.columns):
        ax.bar(x, ancestry_df[ancestry], bottom=bottom,
               color=colors[i % len(colors)], label=ancestry,
               width=bar_width, edgecolor="none")
        bottom += ancestry_df[ancestry]

    # Axis formatting
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Ancestry Proportion", fontsize=12)
    ax.set_xlabel("Individuals", fontsize=12)

    # X-axis label handling
    if not show_labels:
        ax.set_xticks([])
    else:
        ax.set_xticks(x[::max(1, len(ancestry_df)//100)]) # Show every Nth label
        ax.set_xticklabels(
            ancestry_df.index[::max(1, len(ancestry_df)//100)],
            rotation=90, fontsize=6
        )

    # Legend placement
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
              title='Ancestry', frameon=False)

    # Layout adjustment
    plt.tight_layout()

    # Save/show handling
    if save_path:
        save_multi_format(save_path, **kwargs)
        plt.close("all")
    else:
        plt.show()


def plot_ancestry_by_chromosome(
        rf_q: DataFrame, figsize: Tuple[int,int] = (14,6), palette: str = 'Set1',
        save_path: Optional[str] = "chromosome_summary", **kwargs) -> None:
    """
    Plot chromosome-wise ancestry distribution using boxplots.

    Parameters:
    -----------
    rf_q : DataFrame

    figsize : Tuple[int, int], optional
        Figure dimensions in inches (width, height) (default: (14, 6))

    palette : str, optional
        Seaborn color palette name (default: 'Set1')

    save_path : Optional[str], optional
        Base filename for saving plots (without extension). If None, shows
        interactive plot. (default: "chromosome_summary")

    **kwargs : dict
        Additional arguments passed to save_multi_format()

    Example:
    --------
    >>> loci, rf_q, admix = read_rfmix(prefix_path, binary_dir=binary_dir)
    >>> plot_ancestry_by_chromosome(rf_q, dpi=300, bbox_inches="tight")
    """
    # Melt to long-form for Seaborn
    df_long = rf_q.melt(id_vars=['sample_id', 'chrom'], var_name='Ancestry',
                        value_name='Proportion')
    df_long = df_long.to_pandas() if hasattr(df_long, "to_pandas") else df_long
    plt.figure(figsize=figsize)
    sns.boxplot(data=df_long, x='chrom', y='Proportion', hue='Ancestry',
                palette=palette)
    plt.title('Ancestry Proportion per Chromosome')
    plt.ylabel('Ancestry Proportion')
    plt.xlabel('Chromosome')
    plt.legend(title='Ancestry', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    if save_path:
        save_multi_format(save_path, **kwargs)
    else:
        plt.show()


def generate_tagore_bed(
        loci: DataFrame, rf_q: DataFrame, admix: Array, sample_num: int,
        palette: str = "tab10", chunk_size: int = 10_000, min_segment: int = 3,
        verbose: bool = True
) -> DataFrame:
    """
    Generate a BED (Browser Extensible Data) file formatted for TAGORE
    visualization.

    This function processes genomic data and creates a BED file suitable for
    visualization with TAGORE (https://github.com/jordanlab/tagore).

    Parameters:
    -----------
    loci : DataFrame
        A DataFrame containing genomic loci information.
    rf_q : DataFrame
        A DataFrame containing recombination fraction quantiles.
    admix : dask.Array
        An array of admixture proportions.
    sample_num : int
        The sample number to process.
    palette : str, optional
        Colormap name (matplotlib colormap) Default: 'tab10'.
    chunk_size : int, optional
        Size of chunks to process at once (default=10_000).
        Adjust based on available memory.
    min_segment : int, optional
        Minimum length of a segment to consider it a true change (default=3).
    verbose : bool, optional
        If True, print progress information. Defaults to True.

    Returns:
    --------
    DataFrame: A DataFrame in BED format, annotated and ready for TAGORE
               visualization.

    Note:
    -----
        This function relies on several helper functions:
        - admix_to_bed_individual: Converts admixture data to BED format for a
                                   specific individual.
        - _string_to_int: Converts specific columns in the BED DataFrame to
                          integer type (interal function).
        - _annotate_tagore: Adds annotation columns required for TAGORE
                            visualization (internal function).
    """
    pops = get_pops(rf_q)
    bed = admix_to_bed_individual(loci, rf_q, admix, sample_num,
                                  chunk_size, min_segment, verbose)
    sample_cols = bed.columns[3:]
    return _annotate_tagore(bed, sample_cols, pops, palette)


def save_multi_format(filename: str, formats: Tuple[str, ...] = ('png', 'pdf'),
                      **kwargs) -> None:
    """
    Save current figure to multiple file formats.

    Parameters:
    -----------
    filename : str
        Base filename without extension

    formats : Tuple[str, ...], optional
        File extensions to save (default: ('png', 'pdf'))

    **kwargs : dict
        Additional arguments passed to plt.savefig()
    """
    for fmt in formats:
        plt.savefig(f"{filename}.{fmt}", format=fmt, **kwargs)


def _get_global_ancestry(rf_q: DataFrame) -> DataFrame:
    """
    Process raw ancestry data into global proportions.

    Parameters:
    -----------
    rf_q : DataFrame

    Returns:
    --------
    DataFrame
        Processed data with individuals as rows and ancestry proportions as columns
    """
    # Remove chromosome column and group by sample
    return rf_q.drop(columns=['chrom']).groupby('sample_id').mean()


def _annotate_tagore(df: DataFrame, sample_cols: List[str], pops: List[str],
                     palette: str = "tab10") -> DataFrame:
    """
    Annotate a DataFrame with additional columns for visualization purposes.

    This function expands the input DataFrame, adds annotation columns such as
    'feature', 'size', 'color', and 'chrCopy', and renames some columns for
    compatibility with visualization tools.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame to be annotated.
    sample_cols : List(str)
        The name of the column containing sample data.
    pops : List(str)
        List of admixed populations
    palette : str, optional
        Colormap name (matplotlib colormap) Default: 'tab10'.

    Returns:
    --------
    DataFrame: The annotated DataFrame with additional columns.
    """
    # Define a color dictionary to map sample values to palette
    colormap = plt.get_cmap(palette) # Can updated or user defined
    color_dict = {pop: mcolors.to_hex(colormap(i % 10)) for i, pop in enumerate(pops)}
    # Expand the DataFrame using the _expand_dataframe function
    expanded_df = _expand_dataframe(df, sample_cols)
    # Initialize columns for feature and size
    expanded_df["feature"] = 0; expanded_df["size"] = 1
    # Map the sample_cols column to colors using the color_dict
    expanded_df["color"] = expanded_df["sample_name"].map(color_dict)
    # Generate a repeating sequence of 1 and 2
    repeating_sequence = cp.tile(cp.array([1, 2]),
                                 int(cp.ceil(len(expanded_df) / 2)))[:len(expanded_df)]
    # Add the repeating sequence as a new column
    expanded_df['chrCopy'] = repeating_sequence
    # Drop the sample_cols column and rename columns for compatibility
    return expanded_df.drop(["sample_name"], axis=1)\
                      .rename(columns={"chromosome": "#chr", "end": "stop"})


def _expand_dataframe(df: DataFrame, sample_cols: List[str]) -> DataFrame:
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
    ----------
        df (DataFrame): The input dataframe to be expanded.
        sample_name (str): The name of the column to be used for the expansion
                           condition.

    Returns:
    -------
        DataFrame: The expanded and sorted dataframe.
    """
    # Convert to long format
    melted_df = df.melt(id_vars=["chromosome", "start", "end"],
                         value_vars=sample_cols, var_name="sample_ids",
                         value_name="allele_count")
    # Filter non-zero alleles
    melted_df = melted_df[melted_df["allele_count"] > 0]
    # Extract ancestry code from column
    melted_df["sample_name"] = melted_df["sample_ids"].str.extract(r"_([A-Z]+)$")
    # Repeat rows based on allele count
    melted_df = melted_df.loc[melted_df.index.repeat(melted_df['allele_count'])]\
                         .reset_index(drop=True)
    # Select and sort
    result = melted_df[["chromosome", "start", "end", "sample_name"]]
    return result.sort_values(by=['chromosome', 'start', 'sample_name'])\
                 .reset_index(drop=True)
