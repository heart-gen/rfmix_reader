"""
Revision of `_read_flare.py` to work with data generated from
`haptools simgenotype` with population field flag (`--pop_field`).
"""
import numpy as np
from re import sub
from tqdm import tqdm
from glob import glob
from cyvcf2 import VCF
from pathlib import Path
import dask.array as da
from pandas import DataFrame, concat
from typing import List, Tuple, Iterator, Optional
from dask import delayed, compute as dask_compute
from dask.array import Array, concatenate, from_delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import isdir, join, isfile, dirname, basename, exists

from ..utils import _read_file, filter_paths_by_chrom

MISSING = np.uint8(255)

def read_simu(
        vcf_path: str, chunk_size: int = 1_000_000, n_threads: int = 16,
        verbose: bool = True, chrom: Optional[str] = None,
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
    chrom : str, optional
        Restrict parsing to a single chromosome label (with or without ``chr``
        prefix).

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
    fn = _get_vcf_files(vcf_path, chrom=chrom)
    
    # Load loci information
    pbar = tqdm(desc="Mapping loci information", total=len(fn),
                disable=not verbose)
    loci_dfs = _read_file(
        fn,
        lambda f: concat(_read_loci_from_vcf(f, chunk_size),
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
        lambda f: _load_haplotypes_and_global_ancestry(f, chunk_size, n_threads),
        pbar,
    )
    pbar.close()

    # Split into separate lists
    local_chunks, global_dfs = zip(*ancestry_data)

    # Combine global ancestry
    g_anc = concat(global_dfs, ignore_index=True)

    # Combine local ancestry Dask arrays
    local_array = concatenate(local_chunks, axis=0)
    assert local_array.shape[2] == len(g_anc.columns) - 2
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


def _load_haplotypes_and_global_ancestry(
        vcf_file: str, chunk_size: int = 1_000_000, vcf_threads: int = 16,
        dask_chunk: int = 50_000
):
    local_array, ancestries, samples, chrom = \
        _read_haplotypes(vcf_file, chunk_size, vcf_threads, dask_chunk)
    g_anc = _compute_global_from_local(local_array, samples,
                                       chrom, ancestries)
    
    return local_array, g_anc


def _read_haplotypes(
        vcf_file: str, chunk_size: int = 1_000_000, vcf_threads: int = 16,
        dask_chunk: int = 50_000
) -> Tuple[Array, DataFrame]:
    """
    Vectorized local ancestry extraction from VCF with `POP` FORMAT field.
    Uses cyvcf2 region-based pulls (requires tabix index).
    Returns local ancestry Dask array and metadata.
    """
    # Initialize VCF
    _, samples, chrom, chrom_len = _init_vcf(vcf_file, vcf_threads)
    ancestries, mapper = _get_ancestry_labels(vcf_file)

    # Region processor
    def process_region(start: int):
        from cyvcf2 import VCF
        vcf = VCF(vcf_file)
        if vcf_threads and hasattr(vcf, "set_threads"):
            vcf.set_threads(vcf_threads)

        end = min(start + chunk_size - 1, chrom_len)
        region = f"{chrom}:{start}-{end}"
        batch_recs = list(vcf(region))
        vcf.close()
        if not batch_recs:
            return []
        return _process_vectorized_batch(
            batch_recs, ancestries, mapper,
            len(samples), dask_chunk=dask_chunk
        )

    # Thread pool mapping
    starts = range(1, chrom_len + 1, chunk_size)
    local_chunks = []
    with ThreadPoolExecutor(max_workers=vcf_threads) as executor:
        futures = {executor.submit(process_region, start): start for start in starts}
        for future in as_completed(futures):
            try:
                result = future.result()
                local_chunks.extend(result)
            except Exception as e:
                start = futures[future]
                print(f"[ERROR] Chunk at start={start} failed: {e}")
                raise

    # Build final dask.array
    if not local_chunks:
        raise RuntimeError("No valid data extracted from VCF.")

    local_array = concatenate(local_chunks, axis=0)
    return local_array, ancestries, samples, chrom


def _init_vcf(vcf_file, vcf_threads):
    vcf = VCF(vcf_file)
    if vcf_threads and hasattr(vcf, "set_threads"):
        vcf.set_threads(vcf_threads)

    return vcf, vcf.samples, vcf.seqnames[0], vcf.seqlens[0]


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
        return []

    # Collect POP field in one go
    pop_mat = np.array([rec.format("POP") for rec in batch_recs], dtype="U")

    # Vectorized mapping with normalization
    hap_chunk = _map_pop_to_codes(pop_mat, ancestries)

    # Collapse haplotypes -> ancestry axis
    out = np.zeros((n_vars, hap_codes.shape[1], n_anc), dtype=np.uint8)
    for a in range(n_anc):
        out[..., a] = (hap_codes == a).sum(axis=-1)

    # Slice into smaller Dask chunks
    dask_chunks = []
    for start in range(0, n_vars, dask_chunk):
        end = min(start + dask_chunk, n_vars)
        sub_chunk = out[start:end]
        dask_chunks.append(
            from_delayed(delayed(sub_chunk),
                         shape=sub_chunk.shape,
                         dtype=np.uint8)
        )

    return dask_chunks


def _compute_global_from_local(
    local_array: Array, samples: List[str], chrom: str, ancestries: np.ndarray
) -> DataFrame:
    """
    Compute per-sample global ancestry proportions from a local ancestry array.
    """
    n_anc = len(ancestries)

    # Counts per ancestry
    global_counts = [
        (local_array == a).sum(axis=(0, 2))
        for a in range(n_anc)
    ]

    # Stack and normalize using Dask
    counts_da = da.stack(global_counts, axis=1)
    row_sums = counts_da.sum(axis=1, keepdims=True)

    fractions = da.where(
        row_sums > 0,
        counts_da / row_sums,
        0.0
    ).astype(float)

    fractions = fractions.compute()

    # Wrap in DataFrame
    df = DataFrame(fractions, columns=ancestries.tolist())
    df.insert(0, "sample_id", samples)
    df["chrom"] = chrom
    return df


def _normalize_labels(arr: np.ndarray | List[str]) -> np.ndarray:
    """
    Normalize ancestry labels for consistent mapping.
    """
    arr = np.array(arr, dtype="U")
    arr = np.char.strip(arr)
    arr = np.char.upper(arr)
    arr = np.char.replace(arr, " ", "")
    return arr


def _map_pop_to_codes(pop_mat: np.ndarray, ancestries: np.ndarray) -> np.ndarray:
    """
    Map ancestry labels in pop_mat (strings) to numeric codes.
    Uses binary search on sorted ancestry list.
    """
    # Flatten haplotypes
    parts = np.char.partition(pop_mat, ",")
    h0, h1 = parts[:, :, 0], parts[:, :, 2]

    # Normalize both haplotype arrays
    hap = np.stack([h0, h1], axis=-1)
    hap = _normalize_labels(hap)

    # Fast searchsorted lookup
    idx = np.searchsorted(ancestries, hap)
    idx = np.clip(idx, 0, len(ancestries) - 1)
    valid = ancestries[idx] == hap
    codes = np.where(valid, idx.astype(np.uint8), MISSING)

    return codes


def _parse_pop_labels(vcf_file: str, max_records: int = 100) -> List[str]:
    """
    Parse ancestry population labels from a breakpoint (.bp) file
    or from the VCF POP FORMAT field if .bp is missing.

    Ensure VCF is BGZF-compressed and tabix-indexed.
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
        n_scanned = 0
        for rec in vcf:
            try:
                pop = rec.format("POP")
            except Exception as e:
                continue

            if pop is not None:
                flat = np.asarray(pop).astype(str).ravel()
                for entry in flat:
                    if not entry:
                        continue
                    ancestries.update(entry.replace(" ", "").split(","))

            n_scanned += 1
            if n_scanned >= max_records:
                break

    if not ancestries:
        raise ValueError(
            f"No ancestry labels found in .bp file or first {max_records} "
            f"records of VCF: {vcf_file}"
        )

    # Normalize
    ancestries = _normalize_labels(list(ancestries))
    return sorted(set(ancestries))


def _build_mapper(ancestries: List[str]) -> Tuple[np.ndarray, dict[str, np.uint8]]:
    """
    Build fast ancestry lookup: returns sorted ancestry array + dict for labels.
    """
    ancestries = np.array(ancestries, dtype="U")
    mapper = {a: np.uint8(i) for i, a in enumerate(ancestries)}
    return ancestries, mapper


def _get_ancestry_labels(vcf_file):
    ancestries = _parse_pop_labels(vcf_file)
    return _build_mapper(ancestries)


def _get_vcf_files(vcf_path: str, chrom: Optional[str] = None) -> List[str]:
    """
    Resolve a path into a list of ancestry-annotated VCF files.

    Parameters
    ----------
    vcf_path : str
        Path to a directory containing `.vcf` or `.vcf.gz` files.
    chrom : str, optional
        Chromosome label used to filter the results.

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
    vcf_path = Path(vcf_path)

    if vcf_path.is_dir():
        candidates = sorted(vcf_path.glob("*.vcf*"))
    elif vcf_path.is_file() and vcf_path.suffix in {".vcf", ".gz"}:
        candidates = [vcf_path]
    else:
        raise ValueError(
            f"Invalid input: {vcf_path} must be a .vcf, .vcf.gz file, "
            f"or directory containing them."
        )

    # Filter out unwanted files
    vcf_files = []
    for f in candidates:
        suffixes = "".join(f.suffixes)
        if suffixes not in {".vcf", ".vcf.gz"}:
            continue  # skip things like .tbi
        if f.name.endswith("anc.vcf") or f.name.endswith("anc.vcf.gz"):
            continue
        vcf_files.append(f)

    if not vcf_files:
        raise FileNotFoundError(f"No VCF files found in path: {vcf_path}")

    return sorted(filter_paths_by_chrom([str(f) for f in vcf_files], chrom))
