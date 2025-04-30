import seaborn as sns
from dask import config
from dask.array import Array
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Union, List, Optional

from ._loci_bed import admix_to_bed_individual

try:
    import cupy as cp
    from cudf import DataFrame, concat
    config.set({"dataframe.backend": "cudf"})
    config.set({"array.backend": "cupy"})
except ImportError:
    print("Warning: Using CPU!")
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
        palette: Union[str,List[str]] = 'tab20', figsize: Tuple[int,int] = (16,6),
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
        Colormap name (matplotlib colormap) or list of color codes (default: 'tab20')

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
    ancestry_df = _get_global_ancestry(rf_q)
    if hasattr(ancestry_df, "to_pandas"):
        ancestry_df = ancestry_df.to_pandas()
    if sort_by and sort_by in ancestry_df.columns:
        ancestry_df = ancestry_df.sort_values(by=sort_by, ascending=False)
    colors = plt.get_cmap(palette).colors if isinstance(palette, str) else palette
    fig, ax = plt.subplots(figsize=figsize)
    bottom = Series([0] * len(ancestry_df), index=ancestry_df.index)
    for i, col in enumerate(ancestry_df.columns):
        ax.bar(ancestry_df.index, ancestry_df[col], bottom=bottom,
               color=colors[i % len(colors)], label=col)
        bottom += ancestry_df[col]
    ax = ancestry_df.plot(kind='bar', stacked=True, colormap=palette,
                          width=0.85 if len(ancestry_df) < 500 else 1.0)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Ancestry Proportion", fontsize=12)
    ax.set_xlabel("Individuals", fontsize=12)
    if not show_labels:
        ax.set_xticks([])
    else:
        ax.set_xticks(range(len(ancestry_df)))
        ax.set_xticklabels(ancestry_df.index, rotation=90, fontsize=6)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title='Ancestry')
    plt.tight_layout()
    if save_path:
        save_multi_format(save_path, **kwargs)
    else:
        plt.show()


def plot_ancestry_by_chromosome(
        rf_q: DataFrame, figsize: Tuple[int,int] = (14,6), palette: str = 'Set2',
        save_path: Optional[str] = "chromosome_summary", **kwargs) -> None:
    """
    Plot chromosome-wise ancestry distribution using boxplots.

    Parameters:
    -----------
    rf_q : DataFrame

    figsize : Tuple[int, int], optional
        Figure dimensions in inches (width, height) (default: (14, 6))

    palette : str, optional
        Seaborn color palette name (default: 'Set2')

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
    plt.legend(title='Ancestry', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        save_multi_format(save_path, **kwargs)
    else:
        plt.show()


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
        - admix_to_bed_individual: Converts admixture data to BED format for a
                                   specific individual.
        - _string_to_int: Converts specific columns in the BED DataFrame to
                          integer type (interal function).
        - _annotate_tagore: Adds annotation columns required for TAGORE
                            visualization (internal function).
    """
    # Convert admixture data to BED format for the specified sample
    bed = admix_to_bed_individual(loci, rf_q, admix, 0, verbose)
    # Get the name of the sample column (assumed to be the 4th column)
    sample_name = bed.columns[3]
    # Convert string columns to integer type
    bed = _string_to_int(bed, sample_name)
    # Annotate the BED file for TAGORE visualization
    return _annotate_tagore(bed, sample_name)


def _annotate_tagore(df: DataFrame, sample_name: str, pops,
                     colors: str = "tab10") -> DataFrame:
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
    colormap = plt.get_cmap(colors) # Can updated or user defined
    color_dict = {pop: mcolors.to_hex(colormap(i % 10)) for i, pop in enumerate(pops)}
    # Expand the DataFrame using the _expand_dataframe function
    expanded_df = _expand_dataframe(df, sample_name)
    # Initialize columns for feature and size
    expanded_df["feature"] = 0
    expanded_df["size"] = 1
    # Map the sample_name column to colors using the color_dict
    expanded_df["color"] = expanded_df[sample_name].map(color_dict) ## Needs fixing
    # Generate a repeating sequence of 1 and 2
    repeating_sequence = cp.tile(cp.array([1, 2]), ## Check this
                                 int(cp.ceil(len(expanded_df) / 2)))[:len(expanded_df)]
    # Add the repeating sequence as a new column
    expanded_df['chrCopy'] = repeating_sequence
    # Drop the sample_name column and rename columns for compatibility
    return expanded_df.drop([sample_name], axis=1)\
                      .rename(columns={"chromosome": "#chr", "end": "stop"})


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
    ----------
        df (DataFrame): The input dataframe to be expanded.
        sample_name (str): The name of the column to be used for the expansion
                           condition.

    Returns:
    -------
        DataFrame: The expanded and sorted dataframe.
    """
    # Get all columns matching the sample_name prefix
    ancestry_cols = [col for col in df.columns if col.startswith(sample_name)]
    expanded_rows = []
    for _, row in df.iterrows():
        # Process each ancestry column separately
        row_copies = []
        for col in ancestry_cols:
            pop_value = row[col]
            if isinstance(pop_value, int) and pop_value > 1:
                for _ in range(pop_value):
                    row_copy = row.copy()
                    row_copy[col] = 1  # Reduce to single copy
                    row_copies.append(row_copy)
            elif isinstance(pop_value, (list, tuple)):
                for p in pop_value:
                    row_copy = row.copy()
                    row_copy[col] = p
                    row_copies.append(row_copy)
            elif isinstance(pop_value, dict):
                for p, count in pop_value.items():
                    for _ in range(count):
                        row_copy = row.copy()
                        row_copy[col] = p
                        row_copies.append(row_copy)
            else:
                row_copies.append(row.copy())
        # Combine all copies from all ancestry columns
        expanded_rows.extend(row_copies)
        # Create DataFrame and sort
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df.sort_values(
        by=['chromosome', 'start'] + ancestry_cols,
        ascending=[True, True] + [True]*len(ancestry_cols)
    ).reset_index(drop=True)


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


def _viz_dev():
    loci, rf_q, admix = _load_simu_data(3)
    bed = admix_to_bed_individual(loci, rf_q, admix, 13)
    sample_cols = bed.columns[3:]
    bed_df = annotate_tagore(bed, sample_cols)
    return None
