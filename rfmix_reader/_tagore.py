"""
Adapted from `main.py` script in the `tagore` package.
Source: https://github.com/jordanlab/tagore/blob/master/src/tagore/main.py
"""
import re
from pickle import loads
from os import X_OK, path
from cairosvg import svg2png, svg2pdf
from importlib.resources import open_binary
from subprocess import check_output, CalledProcessError

from ._constants import CHROM_SIZES, COORDINATES

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    from cudf import DataFrame
else:
    from pandas import DataFrame

__all__ = [
    "plot_local_ancestry_tagore"
]

def _printif(message: str, verbose: bool):
    """
    Print message if a boolean (e.g. verbose) is true
    """
    if verbose:
        print(message)


def _draw_local_ancestry(
        bed_df: DataFrame, prefix: str, build: str,
        svg_header: str, svg_footer: str, verbose: bool=True
) -> None:
    """
    Create an SVG visualization from a DataFrame of genomic features.

    Parameters:
    -----------
        bed_df (pd.DataFrame): DataFrame with required columns:
            ['#chr', 'start', 'stop', 'feature', 'size', 'color', 'chrCopy']
        prefix (str): Output file prefix.
        build (str): Genome build string ('hg37' or 'hg38').
        svg_header (str): SVG header content.
        svg_footer (str): SVG footer content.
        verbose (bool, optional): If True, print verbose output. Defaults to True.

    Returns:
    --------
        Writes an SVG file named '{prefix}.svg' with the drawn features.

    Raises:
    -------
        IOError, EOFError: If there is an error opening the output file.
        ValueError: If the input DataFrame is missing required columns.
    """
    polygons = ""
    svg_fn = f"{prefix}.svg"

    # Open SVG output file and write header
    try:
        svg_fh = open(svg_fn, "w")
        svg_fh.write(svg_header)
    except (IOError, EOFError) as e:
        print("Error opening output file!")
        raise e

    # Validate required columns
    required_cols = ['#chr','start','stop','feature','size','color','chrCopy']
    if not all(col in bed_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in bed_df.columns]
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}. Missing: {missing}")

    # Process each row in the DataFrame
    bed_df = bed_df.to_pandas() if hasattr(bed_df, "to_pandas") else bed_df

    for line_num, row in bed_df.iterrows():
        try:
            chrm = str(row['#chr']).replace("chr", "")
            start = int(row['start'])
            stop = int(row['stop'])
            feature = int(row['feature'])
            size = float(row['size'])
            col = str(row['color'])
            chrcopy = int(row['chrCopy'])
        except (ValueError, TypeError, KeyError) as e:
            _printif(f"Error processing row {line_num+1}: {e}. Skipping.", verbose)
            continue # Skip rows that fail type conversion

        # Validate size (should be between 0 and 1)
        if size < 0 or size > 1:
            _printif(
                f"Feature size, {size}, on line {line_num+1} unclear. "
                "Please bound the size between 0 (0%) to 1 (100%). Defaulting to 1.",
                verbose
            )
            size = 1

        # Validate color format (hex color starting with '#')
        if not re.match("^#.{6}", col):
            _printif(
                f"Feature color, {col}, on line {line_num+1} unclear. "
                "Please define the color in hex starting with #. Defaulting to #000000.",
                verbose
            )
            col = "#000000"

        # Validate chromosome copy (1 or 2)
        if chrcopy not in [1, 2]:
            _printif(f"Feature chromosome copy, {chrcopy}, on line {line_num+1} unclear. Skipping...", verbose)
            continue

        # Validate chromosome key exists in COORDINATES and CHROM_SIZES
        if chrm not in COORDINATES or chrm not in CHROM_SIZES.get_sizes(build):
            _printif(f"Chromosome {chrm} on line {line_num+1} not recognized. Skipping...",
                     verbose)
            continue

        # Calculate scaled genomic coordinates
        feat_start = start * COORDINATES[chrm]["ht"] / CHROM_SIZES.get_sizes(build)[chrm]
        feat_end = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES.get_sizes(build)[chrm]

        if feature == 0:  # Rectangle
            width = COORDINATES[chrm]["width"] * size / 2
            x_pos = COORDINATES[chrm]["cx"] - width if chrcopy == 1 else COORDINATES[chrm]["cx"]
            y_pos = COORDINATES[chrm]["cy"] + feat_start
            height = feat_end - feat_start
            svg_fh.write(
                f'<rect x="{x_pos}" y="{y_pos}" fill="{col}" width="{width}" height="{height}"/>\n'
            )
        else:
            _printif(f"Feature type, {feature}, unclear on line {line_num+1}. Skipping...",
                     verbose)
            continue

    # Write polygons (triangles) at the end
    svg_fh.write(svg_footer)
    svg_fh.write(polygons)
    svg_fh.write("</svg>")
    svg_fh.close()

    _printif(f"\033[92mSuccessfully created SVG\033[0m", verbose)


def plot_local_ancestry_tagore(
        bed_df: DataFrame, prefix: str, build: str, oformat: str,
        verbose: bool = True, force: bool = False
) -> None:
    """
    Plots local ancestry information from a BED-like DataFrame.

    Parameters:
    -----------
    bed_df : pd.DataFrame
        DataFrame containing BED-like data (e.g., chrom, start, end, sample data).
    prefix : str
        Output file name prefix (without extension).
    build : str
        Genome build ('hg37' or 'hg38').
    oformat : str
        Output image format ('png' or 'pdf'). Defaults to 'png' if unsupported.
    verbose : bool, optional
        If True, print progress messages. Defaults to True.
    force : bool, optional
        If True, overwrite existing output files. Defaults to False.

    Returns:
    --------
        Draw the local ancestry as SVG and converts to a PNG or PDF.

    Raises:
    -------
    ValueError
        If the build is not 'hg37' or 'hg38'.
    FileExistsError
        If the output SVG file already exists and force is False.
    CalledProcessError
        If the SVG conversion fails.
    """
    if build not in ["hg37", "hg38"]:
        raise ValueError(f"\033[91mBuild must be 'hg37' or 'hg38', got '{build}'\033[0m")

    if oformat.lower() not in ["png", "pdf"]:
        print(f"\033[93m{oformat} is not supported. Using PNG instead.\033[0m")
        oformat = "png"

    # Draw local ancestry SVG file
    with open_binary("rfmix_reader", "base.svg.p") as f: ## From tagore
        svg_pkl_data = f.read()

    if not svg_pkl_data:
        raise RuntimeError("\033[91mFailed to load embedded SVG template data.\033[0m")

    svg_header, svg_footer = loads(svg_pkl_data)
    _printif("\033[94mDrawing chromosome ideogram\033[0m", verbose)

    svg_path = f"{prefix}.svg"
    if path.exists(svg_path) and force:
        _draw_local_ancestry(bed_df, prefix, build, svg_header, svg_footer, verbose)
    elif not path.exists(svg_path):
        _draw_local_ancestry(bed_df, prefix, build, svg_header, svg_footer, verbose)
    else:
        _printif(f"\033[93m{svg_path} exists. Skipping drawing. Use `force=True` to overwrite.\033[0m",
                 verbose)

    _printif(f"\033[94mConverting {svg_path} -> {prefix}.{oformat}\033[0m",
             verbose)

    # Convert SVG to PNG or PDF
    try:
        if oformat.lower() == "png":
            svg2png(url=svg_path, write_to=f'{prefix}.png')
        else:
            svg2pdf(url=svg_path, write_to=f'{prefix}.pdf')
    except Exception as convert_err:
        _printif("\033[91mFailed SVG conversion with CairoSVG.\033[0m", verbose)
        raise RuntimeError("SVG conversion failed.") from convert_err
    else:
        _printif(f"\033[92mSuccessfully converted SVG to {oformat.upper()}\033[0m", verbose)
