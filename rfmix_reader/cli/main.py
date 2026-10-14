"""``rfmix-reader`` command line: convert sources to the Zarr cache, inspect it, export for LoGicAl."""
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


def _cmd_export_logical(args: argparse.Namespace) -> int:
    from ..core.api import open_local_ancestry
    from ..ops.logical import to_logical

    ds = open_local_ancestry(args.cache_dir, chrom=args.chrom)
    samples = None
    if args.samples:
        with open(args.samples) as fh:
            samples = [line.strip() for line in fh if line.strip()]
    region = None
    if args.region:
        try:
            lo, hi = args.region.split("-")
            region = (int(lo), int(hi))
        except ValueError:
            raise ValueError("--region must be START-END (1-based, inclusive).")
    chroms = [args.chrom] if args.chrom else ds.la.chromosomes
    for c in chroms:
        path = to_logical(
            ds, args.vcf, args.outdir, chrom=c, samples=samples, method=args.method,
            max_distance=args.max_distance, region=region, snps_only=args.snps_only,
            sites_per_file=args.sites_per_file, windows_per_file=args.windows_per_file,
            apply_phase_swaps=not args.ignore_phase_swaps, write_r_helper=not args.no_r_helper,
            verbose=not args.quiet,
        )
        print(path)
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

    exp = sub.add_parser(
        "export-logical",
        help="write a LoGicAl input bundle (<outdir>/<chrom>/) from a cache and the phased VCF",
    )
    exp.add_argument("cache_dir", help="cache directory (fb with --keep-posteriors for pl/pg/phase_pg)")
    exp.add_argument("vcf", help="the phased VCF/BCF the local ancestry was inferred from")
    exp.add_argument("outdir", help="bundle root; one sub-directory per chromosome")
    exp.add_argument("--chrom", default=None, help="only this chromosome (default: every one in the cache)")
    exp.add_argument("--samples", default=None, help="file with one sample id per line to keep")
    exp.add_argument("--method", choices=["nearest", "stepwise"], default=None,
                     help="site-to-variant mapping (default: nearest for point grids, stepwise for segments)")
    exp.add_argument("--max-distance", type=int, default=None,
                     help="nearest only: drop sites farther than this many bp from a variant")
    exp.add_argument("--region", default=None, help="START-END (1-based, inclusive)")
    exp.add_argument("--snps-only", action="store_true", help="skip indels and other non-SNP records")
    exp.add_argument("--sites-per-file", type=int, default=20_000)
    exp.add_argument("--windows-per-file", type=int, default=None)
    exp.add_argument("--ignore-phase-swaps", action="store_true",
                     help="do not apply a phase_swapped mask from ds.la.phase")
    exp.add_argument("--no-r-helper", action="store_true", help="do not copy logical_run.R into the bundle")
    exp.add_argument("--quiet", action="store_true")
    exp.set_defaults(func=_cmd_export_logical)
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
