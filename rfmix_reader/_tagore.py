"""
Adapted from `main.py` script in the `tagore` package.
Source: https://github.com/jordanlab/tagore/blob/master/src/tagore/main.py
"""
from re import match
from shutil import which
from pickle import loads
from os import X_OK, path
from importlib.resources import open_binary
from subprocess import check_output, CalledProcessError

from ._constants import CHROM_SIZES, COORDINATES

__all__ = [
    "plot_local_ancestry_tagore"
]

def _printif(statement, condition):
    """
    Print statements if a boolean (e.g. verbose) is true
    """
    if condition:
        print(statement)


def _draw_local_ancestry(bed_df, prefix, build, svg_header, svg_footer, verbose=True):
    """
    Create an SVG visualization from a DataFrame of genomic features.

    Args:
        df (pd.DataFrame): DataFrame with columns:
            ['chromosome', 'start', 'stop', 'feature', 'size', 'color', 'chrcopy']
        arguments: Object with attributes:
            - prefix: output file prefix (string)
            - build: genome build string ('hg37' or 'hg38')
            - verbose: boolean flag for verbose output
        svg_header (str): SVG header content
        svg_footer (str): SVG footer content

    Output:
        Writes an SVG file named '{prefix}.svg' with the drawn features.
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
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    # Process each row in the DataFrame
    bed_df = bed_df.pandas() if hasattr(bed_df, "to_pandas") else bed_df
    for line_num, row in bed_df.iterrows():
        chrm = str(row['#chr']).replace("chr", "")
        start = int(row['start'])
        stop = int(row['stop'])
        feature = int(row['feature'])
        size = float(row['size'])
        col = str(row['color'])
        chrcopy = int(row['chrCopy'])
        # Validate size (should be between 0 and 1)
        if size < 0 or size > 1:
            print(
                f"Feature size, {size}, on line {line_num+1} unclear. "
                "Please bound the size between 0 (0%) to 1 (100%). Defaulting to 1."
            )
            size = 1
        # Validate color format (hex color starting with '#')
        if not match("^#.{6}", col):
            print(
                f"Feature color, {col}, on line {line_num+1} unclear. "
                "Please define the color in hex starting with #. Defaulting to #000000."
            )
            col = "#000000"
        # Validate chromosome copy (1 or 2)
        if chrcopy not in [1, 2]:
            print(f"Feature chromosome copy, {chrcopy}, on line {line_num+1} unclear. Skipping...")
            continue
        # Validate chromosome key exists in COORDINATES and CHROM_SIZES
        if chrm not in COORDINATES or chrm not in CHROM_SIZES.get_sizes(build):
            print(f"Chromosome {chrm} on line {line_num+1} not recognized. Skipping...")
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
            print(f"Feature type, {feature}, unclear on line {line_num+1}. Skipping...")
            continue
    # Write polygons (triangles) at the end
    svg_fh.write(svg_footer)
    svg_fh.write(polygons)
    svg_fh.write("</svg>")
    svg_fh.close()
    _printif(f"\033[92mSuccessfully created SVG\033[0m", verbose)


def plot_local_ancestry_tagore(bed_df, prefix, build, oformat, verbose, force):
    if build not in ["hg37", "hg38"]:
        raise ValueError(f"\033[91mBuild must be 'hg37' or 'hg38', got '{build}'\033[0m")
    if which("rsvg-convert", mode=X_OK) is None and which("rsvg", mode=X_OK) is None:
        raise RuntimeError("\033[91mCould not find `rsvg` or `rsvg-convert` in PATH.\033[0m")
    is_rsvg_installed = which("rsvg") is not None
    if oformat not in ["png", "pdf"]:
        print(f"\033[93m{oformat} is not supported. Using PNG instead.\033[0m")
        oformat = "png"
    with open_binary("rfmix_reader", "base.svg.p") as f: ## From tagore
        svg_pkl_data = f.read()
    svg_header, svg_footer = loads(svg_pkl_data)
    _printif("\033[94mDrawing chromosome ideogram\033[0m", verbose)
    if path.exists(f"{prefix}.svg") and not force:
        raise FileExistsError(f"\033[93m'{prefix}.svg' already exists. Use `force=True` to overwrite.\033[0m")
    _draw_local_ancestry(bed_df, prefix, build, svg_header, svg_footer, verbose)
    _printif(
        f"\033[94mConverting {prefix}.svg -> {prefix}.{oformat}\033[0m", verbose
    )
    try:
        if is_rsvg_installed:
            check_output(f"rsvg {prefix}.svg {prefix}.{oformat}", shell=True)
        else:
            check_output(f"rsvg-convert -o {prefix}.{oformat} -f {oformat} {prefix}.svg",
                         shell=True)
    except CalledProcessError as rsvg_e:
        _printif("\033[91mFailed SVG to PNG conversion...\033[0m", verbose)
        raise rsvg_e
    finally:
        _printif(f"\033[92mSuccessfully converted SVG to {oformat.upper()}\033[0m",
                verbose)
