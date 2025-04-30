"""
Adapted from `main.py` script in the `tagore` package.
Source: https://github.com/jordanlab/tagore/blob/master/src/tagore/main.py
"""
import re
import sys

from ._constants import CHROM_SIZES, COORDINATES

def printif(statement, condition):
    """
    Print statements if a boolean (e.g. verbose) is true
    """
    if condition:
        print(statement)


def draw_from_dataframe(df, arguments, svg_header, svg_footer):
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
    svg_fn = f"{arguments.prefix}.svg"

    # Open SVG output file and write header
    try:
        svg_fh = open(svg_fn, "w")
        svg_fh.write(svg_header)
    except (IOError, EOFError) as e:
        print("Error opening output file!")
        raise e

    # Validate required columns
    required_cols = ['chromosome', 'start', 'stop', 'feature', 'size', 'color', 'chrcopy']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Process each row in the DataFrame
    for line_num, row in df.iterrows():
        chrm = str(row['chromosome']).replace("chr", "")
        start = int(row['start'])
        stop = int(row['stop'])
        feature = int(row['feature'])
        size = float(row['size'])
        col = str(row['color'])
        chrcopy = int(row['chrcopy'])

        # Validate size (should be between 0 and 1)
        if size < 0 or size > 1:
            print(
                f"Feature size, {size}, on line {line_num+1} unclear. "
                "Please bound the size between 0 (0%) to 1 (100%). Defaulting to 1."
            )
            size = 1

        # Validate color format (hex color starting with '#')
        if not re.match("^#.{6}", col):
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
        if chrm not in COORDINATES or chrm not in CHROM_SIZES[arguments.build]:
            print(f"Chromosome {chrm} on line {line_num+1} not recognized. Skipping...")
            continue

        # Calculate scaled genomic coordinates
        feat_start = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[arguments.build][chrm]
        feat_end = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[arguments.build][chrm]

        if feature == 0:  # Rectangle
            width = COORDINATES[chrm]["width"] * size / 2
            x_pos = COORDINATES[chrm]["cx"] - width if chrcopy == 1 else COORDINATES[chrm]["cx"]
            y_pos = COORDINATES[chrm]["cy"] + feat_start
            height = feat_end - feat_start

            svg_fh.write(
                f'<rect x="{x_pos}" y="{y_pos}" fill="{col}" width="{width}" height="{height}"/>\n'
            )

        elif feature == 1:  # Circle
            radius = COORDINATES[chrm]["width"] * size / 4
            x_pos = (COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 4 if chrcopy == 1
                     else COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 4)
            y_pos = COORDINATES[chrm]["cy"] + (feat_start + feat_end) / 2

            svg_fh.write(
                f'<circle fill="{col}" cx="{x_pos}" cy="{y_pos}" r="{radius}"/>\n'
            )

        elif feature == 2:  # Triangle
            if chrcopy == 1:
                x_pos = COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 2
                sx_pos = 38.2 * size
            else:
                x_pos = COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 2
                sx_pos = -38.2 * size

            y_pos = COORDINATES[chrm]["cy"] + (feat_start + feat_end) / 2
            sy_pos = 21.5 * size

            polygons += (
                f'<polygon fill="{col}" points="{x_pos-sx_pos},{y_pos-sy_pos} '
                f'{x_pos},{y_pos} {x_pos-sx_pos},{y_pos+sy_pos}"/>\n'
            )

        elif feature == 3:  # Line
            y_pos1 = start * COORDINATES[chrm]["ht"] / CHROM_SIZES[arguments.build][chrm]
            y_pos2 = stop * COORDINATES[chrm]["ht"] / CHROM_SIZES[arguments.build][chrm]
            y_pos = COORDINATES[chrm]["cy"] + (y_pos1 + y_pos2) / 2

            if chrcopy == 1:
                x_pos1 = COORDINATES[chrm]["cx"] - COORDINATES[chrm]["width"] / 2
                x_pos2 = COORDINATES[chrm]["cx"]
            else:
                x_pos1 = COORDINATES[chrm]["cx"]
                x_pos2 = COORDINATES[chrm]["cx"] + COORDINATES[chrm]["width"] / 2

            svg_fh.write(
                f'<line fill="none" stroke="{col}" stroke-miterlimit="10" '
                f'x1="{x_pos1}" y1="{y_pos}" x2="{x_pos2}" y2="{y_pos}"/>\n'
            )

        else:
            print(f"Feature type, {feature}, unclear on line {line_num+1}. Please use 0, 1, 2, or 3. Skipping...")
            continue

    # Write polygons (triangles) at the end
    svg_fh.write(svg_footer)
    svg_fh.write(polygons)
    svg_fh.write("</svg>")
    svg_fh.close()

    printif(f"\033[92mSuccessfully created SVG\033[0m", arguments.verbose)
