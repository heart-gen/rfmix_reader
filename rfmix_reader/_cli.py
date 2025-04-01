import argparse
from . import __version__
from ._utils import create_binaries

def main():
    parser = argparse.ArgumentParser(
        description="Create binary files from RFMix *.fb.tsv files.")
    parser.add_argument(
        "file_path", type=str,
        help="The path used to identify the relevant FB TSV files.")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
                        help="Show the version of the program and exit.")
    parser.add_argument(
        "--binary_dir", type=str, default="./binary_files",
        help="The directory where the binary files will be stored. Defaults to './binary_files'.")

    args = parser.parse_args()
    create_binaries(args.file_path, args.binary_dir)


if __name__ == "__main__":
    main()
