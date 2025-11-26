from pathlib import Path
from typing import List, Optional, Sequence

from bio2zarr.vcf as v2z

def convert_vcf_to_zarr(
    vcf_path: str, out_path: str, *, chunk_length: int = 100_000,
    samples_chunk_size: Optional[int] = None,
    worker_processes: int = 0, verbose: bool = True,
):
    """
    Convert a bgzipped + indexed VCF into a VCF-Zarr store using vcf2zarr/bio2zarr.

    Parameters
    ----------
    vcf_path : str
        Path to the bgzipped + indexed VCF.
    out_path : str
        Output path. Zarr format inferred from extension.
    chunk_length : int
        Genomic chunk size for the output store.
        Larger chunks => better compression, faster sequential reading.
    sample_chunk_size : int, optional
        Passed throught to `sample_chunk_size` in `bio2zarr.vcf.convert`
        (defaults to the library's own behavior if None).
    worker_processes : int, default 0
        Number of worker processes for conversion. 0 lets bio2zarr pick
        a default; you can set this explicitly for your HPC environment.
    verbose : bool
        If True, log progress messages to stdout.
    """
    if verbose:
        print(f"[INFO] Converting VCF to Zarr: {vcf_path} -> {out_path}")

    vcf_path = str(vcf_path); out_path = str(out_path)

    v2z.convert(
        [vcf_path], out_path, variants_chunk_length=chunk_length,
        samples_chunk_size=samples_chunk_size,
        worker_processes=worker_processes, local_alleles=None,
        show_progress=verbose, icf_path=None,
    )

    if verbose:
        print("[INFO] Conversion to Zarr complete.")
        print(f"[DONE] VCF Zarr store ready: {out_path}")


def convert_vcfs_to_zarr(
    vcf_paths: Sequence[str], output_dir: str, *,
    chunk_length: int = 100_000, samples_chunk_size: Optional[int] = None,
    worker_processes: int = 0, verbose: bool = True,
) -> List[str]:
    """
    Convert multiple VCF/BCF files to individual VCF-Zarr stores.

    Parameters
    ----------
    vcf_paths : Sequence[str]
        List of paths to bgzipped + indexed VCF/BCF files.
    output_dir : str
        Directory where the resulting Zarr stores will be written.
    chunk_length : int
        Genomic chunk size for the output store.
    sample_chunk_size : int, optional
        Passed throught to `sample_chunk_size` in `bio2zarr.vcf.convert`
        (defaults to the library's own behavior if None).
    worker_processes : int, default 0
        Number of worker processes for conversion. 0 lets bio2zarr pick
        a default; you can set this explicitly for your HPC environment.
    verbose : bool
        If True, log progress messages to stdout.

    Returns
    -------
    list[str]
        Paths to the generated Zarr stores (one per input VCF).
    """
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
            str(src_path), str(out_path),
            chunk_length=chunk_length,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
            verbose=verbose,
        )
        zarr_paths.append(str(out_path))

    return zarr_paths
