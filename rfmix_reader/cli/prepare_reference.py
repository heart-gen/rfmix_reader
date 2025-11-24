import argparse

from .. import __version__
from ..io.prepare_reference import convert_vcfs_to_zarr


CREATE_REFERENCE_DESCRIPTION = (
    "Convert one or more bgzipped reference VCF/BCF files into Zarr stores."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=CREATE_REFERENCE_DESCRIPTION)
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where the Zarr outputs will be written.",
    )
    parser.add_argument(
        "vcf_paths",
        nargs="+",
        help="Paths to reference VCF/BCF files (bgzipped and indexed).",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="VCF FORMAT/INFO fields to import. Defaults to ['GT'].",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=100_000,
        help="Genomic chunk size for the output Zarr stores (default: 100000).",
    )
    parser.add_argument(
        "--compressor",
        choices=["zstd", "blosc", "lz4", "gzip"],
        default="zstd",
        help="Compression codec to use for the Zarr stores (default: zstd).",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress messages (default: enabled).",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
        help="Show the version of the program and exit.",
    )

    args = parser.parse_args()
    convert_vcfs_to_zarr(
        args.vcf_paths,
        args.output_dir,
        fields=args.fields,
        chunk_length=args.chunk_length,
        compressor=args.compressor,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
