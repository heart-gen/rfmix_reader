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
        vcf_path: str, chunk_size: np.int32 = 1_000_000, n_threads: int = 4,
        verbose: bool = True,
) -> Tuple[DataFrame, DataFrame, Array]:
    """
    Read `haptools simgenotype` generated VCF files into loci, global ancestry,
    and haplotype Dask array.

    Parameters
    ----------
    vcf_path : str
        Path to directory contianing BGZF-compressed VCF files (`.vcf.gz`)
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
        vcf_file: str, chunk_size: np.int32 = 1_000_000
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


def _parse_pop_labels(vcf_file: str) -> List[str]:
    """
    Parse ancestry population labels from a breakpoint (.bp) file
    that corresponds to the given VCF file.

    Parameters
    ----------
    vcf_file : str
        Path to a VCF with a `POP` FORMAT field.

    Returns
    -------
    list of str
        Sorted list of ancestry labels (alphabetical order).

    Raises
    ------
    FileNotFoundError
        If the corresponding breakpoint file does not exist.
    ValueError
        If no ancestry labels could be found in the breakpoint file.
    """
    # Derive .bp file path from VCF path
    vcf_dir = dirname(vcf_file)
    base_name = basename(vcf_file)
    chr_prefix = sub(r"\.vcf\.gz$", "", base_name)
    bp_file = join(vcf_dir, f"{chr_prefix}.bp")

    if not exists(bp_file):
        raise FileNotFoundError(f"Breakpoint file not found: {bp_file}")

    ancestries = set()
    with open(bp_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Sample_"):
                continue  # skip sample headers

            parts = line.split()
            if parts:
                ancestries.add(parts[0])  # ancestry label

    if not ancestries:
        raise ValueError(f"No ancestry labels found in breakpoint file: {bp_file}")

    return sorted(ancestries)


def _load_haplotypes_and_global_ancestry(
        vcf_file: str, chunk_size: int = 50_000, vcf_threads: int = 4
) -> Tuple[Array, DataFrame]:
    """
    Vectorized local/global ancestry extraction from VCF with `POP` FORMAT field.

    Uses cyvcf2 batch pulls instead of per-record loops.
    Keeps local ancestry as lazy Dask array, computes global ancestry eagerly.

    Parameters
    ----------
    vcf_file : str
        Path to ancestry-annotated VCF.
    chunk_size : int, default=10_000
        Number of variant records per processing chunk.

    Returns
    -------
    local_array : dask.array.Array
        Array of shape (variants, samples, ancestries). Each entry indicates
        how many haplotypes (0, 1, or 2) of that ancestry are observed for
        the given variant/sample.
    global_df : DataFrame
        Global ancestry per sample with chromosome label
    """
    # Initialize VCF
    vcf = VCF(vcf_file)
    if vcf_threads and hasattr(vcf, "set_threads"):
        vcf.set_threads(vcf_threads)
        
    samples = vcf.samples
    n_samples = len(samples)
    chrom = vcf.seqnames[0]

    # Extract ancestry labels
    ancestries = sorted(set(_parse_pop_labels(vcf_file)))
    n_ancestries = len(ancestries)
    mapper = _build_mapper(ancestries)

    # Init global counters
    delayed_chunks = []
    global_counts = np.zeros((n_samples, n_ancestries), dtype=np.int64)
    total_alleles = 0
    batch, blen = [], 0

    def _make_delayed(records_local):
        return delayed(_finalize_batch_compact)(records_local, ancestries, mapper, n_samples)
    
    while True:
        try:
            rec = next(vcf)
        except StopIteration:
            break
        batch.append(rec); blen += 1
        if blen >= chunk_size:
            d = _make_delayed(batch)
            codes_d = delayed(lambda x: x[0])(d)
            g_d     = delayed(lambda x: (x[1], x[2]))(d)
            n_vars_local = blen
            delayed_chunks.append((
                from_delayed(codes_d, shape=(n_vars_local, n_samples, n_ancestries), dtype=np.int8),
                g_d
            ))
            batch, blen = [], 0

    # Flush remainder
    if blen > 0:
        d = _make_delayed(batch)
        codes_d = delayed(lambda x: x[0])(d)
        g_d     = delayed(lambda x: (x[1], x[2]))(d)
        delayed_chunks.append((
            from_delayed(codes_d, shape=(blen, n_samples, n_ancestries), dtype=np.int8),
            g_d
        ))

    # Build data
    local_array = concatenate([lc for (lc, _) in delayed_chunks], axis=0)
    global_pairs = [gd for (_, gd) in delayed_chunks]

    # Aggregate global fractions
    for g_inc, alleles_inc in dask_compute(*global_pairs):
        global_counts += g_inc
        total_alleles += alleles_inc

    # Normalize global ancestry
    fractions = (global_counts / max(total_alleles, 1)).astype(float)
    fractions = np.nan_to_num(fractions, nan=0.0)

    # Build dataframe
    global_df = DataFrame(fractions, columns=ancestries)
    global_df.insert(0, "sample_id", samples)
    global_df["chrom"] = chrom

    return local_array, global_df


def _finalize_batch_compact(batch_recs, ancestries, mapper, n_samples):
    """Return (codes_chunk uint8, global_counts int64[ n_samples, n_ancestries ], alleles_in_batch)"""
    n_anc = len(ancestries)
    n_vars = len(batch_recs)
    if n_vars == 0:
        return (np.empty((0, n_samples, 2), dtype=np.uint8),
                np.zeros((n_samples, n_anc), dtype=np.int64),
                0)

    # Gather POP as a single (n_vars, n_samples) unicode array
    pop_flat = np.empty(n_vars * n_samples, dtype=object)
    idx = 0
    for rec in batch_recs:
        pop = rec.format("POP")  # len = n_samples; bytes or str like "AFR,EUR"
        if pop is None:
            pop_flat[idx:idx+n_samples] = ""
        else:
            pf = np.asarray(pop)
            pop_flat[idx:idx+n_samples] = pf.astype('U')
        idx += n_samples

    pop_mat = pop_flat.reshape(n_vars, n_samples).astype('U', copy=False)
    codes_chunk = _split_pop_to_codes(pop_mat, mapper)  # (n_vars, n_samples, 2), uint8

    # Global counts by sample via bincount on codes (ignore 255)
    gc = np.zeros((n_samples, n_anc), dtype=np.int64)
    # reshape to (n_vars*n_samples*2,)
    flat = codes_chunk.reshape(-1)
    mask = flat != MISSING
    valid = flat[mask].astype(np.int64, copy=False)
    bc = np.bincount(valid, minlength=n_anc)  # total over all samples
    for s in range(n_samples):
        vs = codes_chunk[:, s, :].reshape(-1)
        m = vs != MISSING
        if m.any():
            gc[s] = np.bincount(vs[m].astype(np.int64), minlength=n_anc)

    alleles = 2 * n_vars
    return codes_chunk, gc, alleles


def _split_pop_to_codes(pop_mat_U, mapper):
    """
    pop_mat_U: (n_vars, n_samples) np.ndarray with unicode like "AFR,EUR"
    Returns: codes uint8 of shape (n_vars, n_samples, 2) with MISSING=255
    """
    parts = np.char.partition(pop_mat_U, ",")  # (n_vars, n_samples, 3): [h0, ',', h1]
    h0 = parts[:, :, 0]
    h1 = parts[:, :, 2]

    # Mapper that returns 255 when missing/unknown
    def map_vec(arr):
        # vectorized dict.get; otypes must be [np.uint8]
        f = np.vectorize(lambda x: mapper.get(x, MISSING), otypes=[np.uint8])
        return f(arr)

    c0 = map_vec(h0)
    c1 = map_vec(h1)
    return np.stack([c0, c1], axis=-1)


def _build_mapper(ancestries):
    # Example: {"AFR":0, "EUR":1, ...}
    return {a: np.uint8(i) for i, a in enumerate(ancestries)}


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
