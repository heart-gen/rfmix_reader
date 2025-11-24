from pathlib import Path
from typing import List, Optional, Sequence

from sgkit.io.vcf.vcf_reader import vcf_to_zarr

def convert_vcf_to_zarr(
    vcf_path: str, out_path: str, *, fields: Optional[List[str]] = None,
    chunk_length: int = 100_000, compressor: str = "zstd",
    verbose: bool = True,
):
    """
    Convert a VCF (bgzipped + indexed) into Zarr or Parquet for
    permanent 50–200x faster downstream access.

    Parameters
    ----------
    vcf_path : str
        Path to the bgzipped + indexed VCF.
    out_path : str
        Output path. Zarr format inferred from extension.
    fields : list[str], optional
        VCF FORMAT/INFO fields to import (default common fields).
        Use `['GT']` for genotypes only.
    chunk_length : int
        Genomic chunk size for the output store.
        Larger chunks => better compression, faster sequential reading.
    compressor : str
        One of: "zstd", "blosc", "lz4", "gzip".
    verbose : bool
        If True, log progress messages to stdout.
    """
    if fields is None:
        fields = ["GT"]

    if verbose:
        print(f"[INFO] Converting VCF to Zarr: {out_path}")
    ds = vcf_to_zarr(
        path=vcf_path, output=out_path, fields=fields,
        chunk_length=chunk_length, compressor=compressor,
        temp_chunk_store=None, overwrite=True,
    )
    if verbose:
        print("[INFO] Conversion to Zarr complete.")
        print(f"[DONE] Zarr store ready: {out_path}")


def convert_vcfs_to_zarr(
    vcf_paths: Sequence[str],
    output_dir: str,
    *,
    fields: Optional[List[str]] = None,
    chunk_length: int = 100_000,
    compressor: str = "zstd",
    verbose: bool = True,
) -> List[str]:
    """
    Convert multiple reference VCF/BCF files to Zarr stores.

    Parameters
    ----------
    vcf_paths : Sequence[str]
        List of paths to bgzipped + indexed VCF/BCF files.
    output_dir : str
        Directory where the resulting Zarr stores will be written.
    fields : list[str], optional
        VCF FORMAT/INFO fields to import (default common fields).
        Use `['GT']` for genotypes only.
    chunk_length : int
        Genomic chunk size for the output store.
    compressor : str
        One of: "zstd", "blosc", "lz4", "gzip".
    verbose : bool
        If True, log progress messages to stdout.

    Returns
    -------
    list[str]
        Paths to the generated Zarr stores.
    """

    if fields is None:
        fields = ["GT"]

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    zarr_paths: List[str] = []
    for vcf_path in vcf_paths:
        src_path = Path(vcf_path)
        name = src_path.name
        if name.endswith(".vcf.gz"):
            name = name[:-7]
        elif name.endswith(".vcf.bgz"):
            name = name[:-8]
        elif name.endswith(".vcf"):
            name = name[:-4]
        elif name.endswith(".bcf"):
            name = name[:-4]

        out_path = out_dir_path / f"{name}.zarr"
        convert_vcf_to_zarr(
            str(src_path),
            str(out_path),
            fields=fields,
            chunk_length=chunk_length,
            compressor=compressor,
            verbose=verbose,
        )
        zarr_paths.append(str(out_path))

    return zarr_paths
