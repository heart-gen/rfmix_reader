"""
Revision of `_read_flare.py` to work with data generated from
`haptools simgenotype` with population field flag (`--pop_field`).
"""
import itertools
from tqdm import tqdm
from glob import glob
from cyvcf2 import VCF
from dask import delayed
from re import search, sub
from numpy import int32, empty
from typing import List, Tuple, Iterator, Dict
from dask.array import Array, concatenate, from_delayed, stack
from os.path import isdir, join, isfile, dirname, basename, exists

from ._utils import set_gpu_environment, _read_file

try:
    from torch.cuda import is_available as gpu_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def gpu_available(): return False

if gpu_available():
    from cudf import DataFrame, concat
    from cupy import (
        zeros,
        asnumpy,
        nan_to_num,
        int64 as xp_int64,
        asarray, unique, int8,
    )
else:
    from pandas import DataFrame, concat
    from numpy import (
        zeros,
        nan_to_num,
        int64 as xp_int64,
        asarray, unique, int8,
    )
    def asnumpy(x): return x

__all__ = ["read_simu"]

def read_simu(
        vcf_path: str, chunk_size: int32 = 1_000_000, verbose: bool = True,
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
    # Device information
    if verbose and gpu_available():
        set_gpu_environment()

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
        lambda f: _load_haplotypes_and_global_ancestry(f, chunk_size // 100),
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
        vcf_file: str, chunk_size: int32 = 1_000_000
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
    vcf_file: str, chunk_size: int = 50_000
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
    samples = vcf.samples
    n_samples = len(samples)
    n_variants = vcf.num_records

    # Extract ancestry labels
    ancestries = sorted(set(_parse_pop_labels(vcf_file)))
    n_ancestries = len(ancestries)
    anc_index = {label: i for i, label in enumerate(ancestries)}

    # Init global counters
    global_counts = zeros((n_samples, n_ancestries), dtype=xp_int64)
    total_alleles = zeros(n_samples, dtype=xp_int64)
    
    delayed_chunks = []

    # Helper to process batch
    def process_batch(start, end):
        """Process a slice of variants [start:end) into local ancestry array."""
        try:
            # Pull POP FORMAT in one call: shape (variants, samples)
            pop_matrix = vcf[start:end].format("POP")
        except Exception as e:
            raise RuntimeError(f"Error reading POP field in {vcf_file}: {e}")

        n_vars = end - start
        local_chunk = zeros((n_vars, n_samples, n_ancestries), dtype=int8)

        # Split haplotype labels vectorized
        hap0 = empty((n_vars, n_samples), dtype=object) # This is NumPy for string
        hap1 = empty((n_vars, n_samples), dtype=object)
        for i in range(n_vars):
            for j in range(n_samples):
                raw = pop_matrix[i][j]
                if raw:
                    parts = raw.split(",")
                    hap0[i, j] = parts[0].strip()
                    hap1[i, j] = parts[1].strip()

        # Map hap labels -> indices
        idx0 = array([[anc_index.get(x, -1) for x in row] for row in hap0])
        idx1 = array([[anc_index.get(x, -1) for x in row] for row in hap1])

        # Fill local ancestry counts
        for a, idx in anc_index.items():
            local_chunk[:, :, idx] += (idx0 == idx).astype(int8)
            local_chunk[:, :, idx] += (idx1 == idx).astype(int8)

        # Update global counts deterministically
        global_counts[:] += local_chunk.sum(axis=0)
        total_alleles[:] += 2 * n_vars

        return local_chunk

    # Chunk loop
    for start, end in _chunk_intervals(n_variants, chunk_size):
        delayed_chunks.append(
            from_delayed(
                delayed(process_batch)(start, end),
                shape=(end - start, n_samples, n_ancestries),
                dtype="i1",
            )
        )

    local_array = concatenate(delayed_chunks, axis=0)
    
    # Normalize global ancestry
    fractions = global_counts / total_alleles[:, None]
    fractions = asnumpy(nan_to_num(fractions, nan=0.0))

    # Build dataframe
    global_df = DataFrame(fractions, columns=ancestries)
    global_df.insert(0, "sample_id", samples)

    m = search(r'chr[\w]+', vcf_file)
    global_df["chrom"] = m.group(0) if m else None

    return local_array, global_df


def _chunk_intervals(n_variants: int, size: int):
    """Yield (start, end) variant index intervals of length `size`."""
    for start in range(0, n_variants, size):
        yield start, min(start + size, n_variants)


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
