import argparse

from .. import __version__


def _invoke_create_binaries(file_path: str, binary_dir: str) -> None:
    from ..utils import create_binaries as create_binaries_func

    create_binaries_func(file_path, binary_dir)


create_binaries = _invoke_create_binaries


def main() -> None:
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
