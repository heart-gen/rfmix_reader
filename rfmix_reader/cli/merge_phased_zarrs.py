import argparse

from .. import __version__
from ..processing.phase import merge_phased_zarrs

MERGE_DESCRIPTION = (
    "Merge per-chromosome phased Zarr outputs produced by phase_rfmix_chromosome_to_zarr."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=MERGE_DESCRIPTION)
    parser.add_argument(
        "output_path",
        type=str,
        help="Destination path for the merged Zarr store.",
    )
    parser.add_argument(
        "chrom_zarr_paths",
        nargs="+",
        help="Paths to per-chromosome phased Zarr stores to merge.",
    )
    parser.add_argument(
        "--sort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sort by chromosome and variant_position after concatenation "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version of the program and exit.",
    )

    args = parser.parse_args()
    merge_phased_zarrs(args.chrom_zarr_paths, args.output_path, sort=args.sort)


if __name__ == "__main__":
    main()
