"""``rfmix-reader`` command line: convert sources to the Zarr cache, inspect it."""
from __future__ import annotations

import argparse
import logging
import sys

from .. import __version__


def _cmd_convert(args: argparse.Namespace) -> int:
    from ..core.api import convert

    paths = convert(
        args.path, args.format, args.cache_dir, chrom=args.chrom,
        keep_posteriors=args.keep_posteriors, overwrite=args.overwrite,
        chunk_rows=args.chunk_rows, n_threads=args.threads, verbose=not args.quiet,
    )
    for p in paths:
        print(p)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from ..core.api import open_local_ancestry

    ds = open_local_ancestry(args.cache_dir, chrom=args.chrom)
    la = ds.la
    print(f"variants:   {la.n_variants}")
    print(f"samples:    {la.n_samples}")
    print(f"ancestries: {', '.join(la.ancestries)}")
    print(f"chromosomes: {', '.join(la.chromosomes)}")
    print(f"posterior:  {'yes' if la.posterior is not None else 'no'}")
    print(f"source:     {ds.attrs.get('source_format', '?')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfmix-reader",
        description="Convert local-ancestry outputs to a lazy Zarr cache and inspect it.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="convert a source into <cache_dir>/<chrom>.zarr stores")
    conv.add_argument("format", choices=["msp", "fb", "flare", "haptools"])
    conv.add_argument("path", help="directory, file or path prefix of the source files")
    conv.add_argument("cache_dir", help="output directory for the Zarr stores")
    conv.add_argument("--chrom", default=None, help="only this chromosome")
    conv.add_argument("--keep-posteriors", action="store_true",
                      help="fb only: also store the float32 posteriors")
    conv.add_argument("--overwrite", action="store_true", help="rebuild existing stores")
    conv.add_argument("--chunk-rows", type=int, default=10_000)
    conv.add_argument("--threads", type=int, default=4, help="haptools region threads")
    conv.add_argument("--quiet", action="store_true")
    conv.set_defaults(func=_cmd_convert)

    info = sub.add_parser("info", help="summarise a cache directory")
    info.add_argument("cache_dir")
    info.add_argument("--chrom", default=None)
    info.set_defaults(func=_cmd_info)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    try:
        code = args.func(args)
    except Exception as e:  # clean CLI failure
        parser.exit(1, f"error: {e}\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
