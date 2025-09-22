"""
Revision of `_read_flare.py` to work with data generated from
`haptools simgenotype` with population field flag (`--pop_field`).
"""
import numpy as np
from re import sub
from tqdm import tqdm
from glob import glob
from cyvcf2 import VCF
from pandas import DataFrame, concat
from typing import List, Tuple, Iterator
from dask import delayed, compute as dask_compute
from dask.array import Array, concatenate, from_delayed
from os.path import isdir, join, isfile, dirname, basename, exists

from ._utils import _read_file

__all__ = ["read_simu"]

MISSING = np.uint8(255)

def read_simu(
        vcf_path: str, chunk_size: int = 1_000_000, n_threads: int = 4,
        verbose: bool = True,
) -> Tuple[DataFrame, DataFrame, Array]:
    """
    Read `haptools simgenotype` generated VCF files into loci, global ancestry,
    and haplotype Dask array.

    Parameters
    ----------
    vcf_path : str
        Path to directory containing BGZF-compressed VCF files (`.vcf.gz`)
        (e.g., one per chromosome).
    chunk_size : int, default=1_000_000
        Number of variant records to process per chunk when reading. Smaller
        values reduce memory footprint, at the cost of more I/O.
    verbose : bool, default=True
        If True, show progress bars during parsing.

    Returns
    -------
    loci_df : :class:`DataFrame`
        Chromosome, physical position, and sequential index of all variants.
        Columns: ['chromosome', 'physical_position', 'i'].
    g_anc : :class:`DataFrame`
        Per-sample global ancestry proportions for each chromosome.
        Columns: ['sample_id', <ancestry labels...>, 'chrom'].
    local_array : :class:`dask.array.Array`
        Local ancestry counts with shape (variants, samples, ancestries).
        The last axis is ordered alphabetically by ancestry label, ensuring
        compatibility with RFMix-style conventions.
    """
    # Get VCF file prefixes
    fn = _get_vcf_files(vcf_path)

    # Load loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn),
                disable=not verbose)
    loci_dfs = _read_file(
        fn,
        lambda f: concat(list(_read_loci_from_vcf(f, chunk_size)),
                         ignore_index=True),
        pbar,
    )
    pbar.close()

    index_offset = 0
    for df in loci_dfs: # Modify in-place
        df["i"] = range(index_offset, index_offset + df.shape[0])
        index_offset += df.shape[0]
    loci_df = concat(loci_dfs, ignore_index=True)

    # Load local and global ancestry per chromosome
    pbar = tqdm(desc="Mapping ancestry data", total=len(fn),
                disable=not verbose)
    ancestry_data = _read_file(
        fn,
        lambda f: _load_haplotypes_and_global_ancestry(f, chunk_size // 100, n_threads),
        pbar,
    )
    pbar.close()

    # Split into separate lists
    local_chunks, global_dfs = zip(*ancestry_data)

    # Combine global ancestry
    g_anc = concat(global_dfs, ignore_index=True)

    # Combine local ancestry Dask arrays
    local_array = concatenate(local_chunks, axis=0)

    return loci_df, g_anc, local_array


def _read_loci_from_vcf(
        vcf_file: str, chunk_size: int = 1_000_000
) -> Iterator[DataFrame]:
    """
    Extract loci information (chromosome and position) from a VCF file.

    Parameters
    ----------
    vcf_file : str
        Path to a BGZF-compressed, tabix-indexed VCF file.
    chunk_size : int, default=1_000_000
        Number of variant records per yielded chunk.

    Yields
    ------
    DataFrame
        A DataFrame containing 'chromosome' and 'physical_position' for each chunk.
    """
    vcf = VCF(vcf_file)
    loci = []
    for rec in vcf:
        loci.append({
            'chromosome': rec.CHROM,
            'physical_position': rec.POS
        })
        if len(loci) >= chunk_size:
            yield DataFrame(loci)
            loci = []

    if loci:
        yield DataFrame(loci)


def _parse_pop_labels(vcf_file: str, max_records: int = 100) -> List[str]:
    """
    Parse ancestry population labels from a breakpoint (.bp) file
    or from the VCF POP FORMAT field if .bp is missing.
    """
    # Derive .bp file path from VCF path
    vcf_dir = dirname(vcf_file)
    base_name = basename(vcf_file)
    chr_prefix = sub(r"\.vcf\.gz$", "", base_name)
    bp_file = join(vcf_dir, f"{chr_prefix}.bp")

    ancestries = set()

    if exists(bp_file):
        # Primary: read from .bp file (faster)
        with open(bp_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Sample_"):
                    continue

                parts = line.split()
                if parts:
                    ancestries.add(parts[0])
    else:
        # Fallback
        vcf = VCF(vcf_file)
        for i, rec in enumerate(vcf):
            pop = rec.format("POP")
            if pop is not None:
                flat = np.asarray(pop).astype(str).ravel()
                for entry in flat:
                    if not entry:
                        continue
                    ancestries.update(entry.replace(" ", "").split(","))
            if i + 1 >= max_records:
                break

    if not ancestries:
        raise ValueError(
            f"No ancestry labels found in .bp file or first {max_records} "
            f"records of VCF: {vcf_file}"
        )

    # Normalize
    ancestries = _normalize_labels(list(ancestries))
    return sorted(set(ancestries))


def _load_haplotypes_and_global_ancestry(
        vcf_file: str, chunk_size: int = 50_000, vcf_threads: int = 4,
        dask_chunk: int = 50_000
) -> Tuple[Array, DataFrame]:
    """
    Vectorized local/global ancestry extraction from VCF with `POP` FORMAT field.
    Internally processes large chunks (fast, vectorized), then yields smaller
    slices to Dask (memory-friendly).

    Uses cyvcf2 batch pulls instead of per-record loops.
    Keeps local ancestry as lazy Dask array, computes global ancestry eagerly.
    """
    # Initialize VCF
    vcf, samples, chrom = _init_vcf(vcf_file, vcf_threads)
    ancestries, mapper = _get_ancestry_labels(vcf_file)
    n_samples, n_anc = len(samples), len(ancestries)

    # Global ancestry accumulators
    global_counts = np.zeros((n_samples, n_anc), np.int64)
    total_alleles = 0
    local_chunks = []

    batch = []
    for rec in vcf:
        batch.append(rec)
        if len(batch) >= chunk_size:
            new_chunks, gc, alleles = _process_vectorized_batch(
                batch, ancestries, mapper, n_samples,
                dask_chunk=dask_chunk
            )
            local_chunks.extend(new_chunks)
            global_counts += gc
            total_alleles += alleles
            batch = []

    if batch:
        new_chunks, gc, alleles = _process_vectorized_batch(
            batch, ancestries, mapper, n_samples,
            dask_chunk=dask_chunk
        )
        local_chunks.extend(new_chunks)
        global_counts += gc
        total_alleles += alleles

    # Build final dask.array
    local_array = concatenate(local_chunks, axis=0)

    # Global ancestry proportions
    global_df = _finalize_global_ancestry(
        samples, chrom, ancestries, global_counts, total_alleles
    )
    return local_array, global_df


def _init_vcf(vcf_file, vcf_threads):
    vcf = VCF(vcf_file)
    if vcf_threads and hasattr(vcf, "set_threads"):
        vcf.set_threads(vcf_threads)
    return vcf, vcf.samples, vcf.seqnames[0]


def _get_ancestry_labels(vcf_file):
    ancestries = _parse_pop_labels(vcf_file)
    return _build_mapper(ancestries)


def _process_vectorized_batch(
    batch_recs, ancestries, mapper, n_samples, dask_chunk=50_000
):
    """
    Vectorized ancestry extraction for a large batch, then slice
    into smaller Dask chunks to bound memory.
    """
    n_vars = len(batch_recs)
    n_anc = len(ancestries)
    if n_vars == 0:
        return [], np.zeros((n_samples, n_anc), dtype=np.int64), 0

    # Collect POP field in one go
    pop_mat = np.array([rec.format("POP") for rec in batch_recs], dtype="U")

    # Vectorized mapping with normalization
    codes_chunk = _map_pop_to_codes(pop_mat, ancestries)

    # Global ancestry counts (vectorized)
    gc = np.zeros((n_samples, n_anc), dtype=np.int64)
    for a in range(n_anc):
        gc[:, a] = (codes_chunk == a).sum(axis=(0, 2))

    alleles = 2 * n_vars

    # Slice into smaller Dask chunks
    dask_chunks = []
    for start in range(0, n_vars, dask_chunk):
        end = min(start + dask_chunk, n_vars)
        sub = codes_chunk[start:end]  # view slice
        dask_chunks.append(
            from_delayed(
                delayed(lambda x: x)(sub),
                shape=sub.shape,
                dtype=np.uint8
            )
        )

    return dask_chunks, gc, alleles


def _normalize_labels(arr: np.ndarray | list[str]) -> np.ndarray:
    """
    Normalize ancestry labels for consistent mapping.
    """
    arr = np.array(arr, dtype="U")
    arr = np.char.strip(arr)
    arr = np.char.upper(arr)
    arr = np.char.replace(arr, " ", "")
    return arr


def _finalize_global_ancestry(samples, chrom, ancestries, global_counts, total_alleles):
    row_sums = global_counts.sum(axis=1, keepdims=True)
    fractions = np.divide(
        global_counts,
        np.maximum(row_sums, 1),  # avoid div/0
        where=row_sums > 0
    ).astype(float)
    fractions = np.nan_to_num(fractions, nan=0.0)

    df = DataFrame(fractions, columns=ancestries.tolist())
    df.insert(0, "sample_id", samples)
    df["chrom"] = chrom
    return df


def _map_pop_to_codes(pop_mat: np.ndarray, ancestries: np.ndarray) -> np.ndarray:
    """
    Map ancestry labels in pop_mat (strings) to numeric codes.
    Uses binary search on sorted ancestry list.
    """
    # Flatten haplotypes
    parts = np.char.partition(pop_mat, ",")
    h0, h1 = parts[:, :, 0], parts[:, :, 2]

    # Normalize both haplotype arrays
    h0 = _normalize_labels(h0)
    h1 = _normalize_labels(h1)
    hap = np.stack([h0, h1], axis=-1)

    # Fast searchsorted lookup
    idx = np.searchsorted(ancestries, hap)
    idx = np.clip(idx, 0, len(ancestries)-1)
    valid = ancestries[idx] == hap
    codes = np.where(valid, idx.astype(np.uint8), MISSING)

    return codes


def _build_mapper(ancestries: list[str]) -> tuple[np.ndarray, dict[str, np.uint8]]:
    """
    Build fast ancestry lookup: returns sorted ancestry array + dict for labels.
    """
    ancestries = np.array(ancestries, dtype="U")
    mapper = {a: np.uint8(i) for i, a in enumerate(ancestries)}
    return ancestries, mapper


def _get_vcf_files(vcf_path: str) -> List[str]:
    """
    Resolve a path into a list of ancestry-annotated VCF files.

    Parameters
    ----------
    vcf_path : str
        Path to a directory containing `.vcf.gz` files.

    Returns
    -------
    list of str
        Sorted list of VCF file paths.

    Raises
    ------
    ValueError
        If `vcf_path` is not a valid file or directory.
    FileNotFoundError
        If no VCF files matching the pattern are found.
    """
    if isdir(vcf_path):
        vcf_files = sorted(
            f for f in glob(join(vcf_path, "*.vcf.gz"))
            if not f.endswith("anc.vcf.gz")
        )
    elif isfile(vcf_path) and vcf_path.endswith(".vcf.gz"):
        if vcf_path.endswith("anc.vcf.gz"):
            vcf_files = []
        else:
            vcf_files = [vcf_path]
    else:
        raise ValueError(f"Invalid input: {vcf_path} must be a .vcf.gz file or directory containing them.")

    if not vcf_files:
        raise FileNotFoundError(f"No VCF files found in path: {vcf_path}")

    return vcf_files
