"""
Revision of `_read_flare.py` to work with data generated from
`haptools simgenotype` with population field flag (`--pop_field`).
"""
from re import search
from tqdm import tqdm
from glob import glob
from cyvcf2 import VCF
from numpy import int32
from dask import delayed
from os.path import isdir, join, isfile
from typing import List, Tuple, Iterator
from dask.array import Array, concatenate, from_delayed, stack

from ._utils import set_gpu_environment, _read_file

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

if is_available():
    from cupy import zeros, int8
    from cudf import DataFrame, concat
else:
    from numpy import zeros, int8
    from pandas import DataFrame, concat

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
    if verbose and is_available():
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

    # Load global ancestry per chromosome
    pbar = tqdm(desc="Mapping global ancestry files", total=len(fn),
                disable=not verbose)
    g_anc = _read_file(
        fn,
        lambda f: _calculate_global_ancestry_from_pop(f),
        pbar,
    )
    pbar.close()
    g_anc = concat(g_anc, ignore_index=True)

    # Loading local ancestry by loci
    pbar = tqdm(desc="Mapping local ancestry files", total=len(fn),
                disable=not verbose)
    local_array = _read_file(
        fn,
        lambda f: _load_haplotypes_from_pop(f, chunk_size // 100),
        pbar,
    )
    pbar.close()
    local_array = concatenate(local_array, axis=0)

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
    Parse ancestry population labels from the `POP` field of a VCF.

    This function inspects the first record with non-empty `POP` annotations
    and extracts all unique ancestry labels.

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
    ValueError
        If no `POP` field is found in the file.
    """
    vcf = VCF(vcf_file)
    for rec in vcf:
        for sample_field in rec.format("POP"):
            if sample_field:
                parts = sample_field.split(",")
                return sorted(set(parts))
    raise ValueError("No POP field found in VCF.")


def _load_haplotypes_from_pop(vcf_file: str, chunk_size: int = 10_000) -> Array:
    """
    Build a local ancestry tensor from the `POP` FORMAT field.

    Each allele for each sample is labeled by ancestry in the VCF.
    This function constructs a Dask array of counts per ancestry.

    Parameters
    ----------
    vcf_file : str
        Path to ancestry-annotated VCF.
    chunk_size : int, default=10_000
        Number of variant records per processing chunk.

    Returns
    -------
    dask.array.Array
        Array of shape (variants, samples, ancestries). Each entry indicates
        how many haplotypes (0, 1, or 2) of that ancestry are observed for
        the given variant/sample.
    """
    vcf = VCF(vcf_file)
    samples = vcf.samples
    n_samples = len(samples)
    ancestries = sorted(_parse_pop_labels(vcf_file))
    anc_index = {label: i for i, label in enumerate(ancestries)}
    n_ancestries = len(ancestries)

    def process_chunk(records):
        chunk_len = len(records)
        counts = zeros((chunk_len, n_samples, n_ancestries), dtype=int8)

        for i, rec in enumerate(records):
            pop_fields = rec.format("POP")
            for s_idx, pop_string in enumerate(pop_fields):
                if pop_string:
                    haps = [h.strip() for h in pop_string.split(",") if h.strip()]
                    for h in haps:
                        counts[i, s_idx, anc_index[h]] += 1
        return counts

    buffer = []
    delayed_chunks = []
    for rec in vcf:
        buffer.append(rec)
        if len(buffer) == chunk_size:
            delayed_chunks.append(delayed(process_chunk)(buffer))
            buffer = []

    if buffer:
        delayed_chunks.append(delayed(process_chunk)(buffer))

    arrays = [from_delayed(d, shape=(None, n_samples, n_ancestries), dtype=int8)
              for d in delayed_chunks]
    stacked = concatenate(arrays, axis=0)
    return stacked


def _calculate_global_ancestry_from_pop(vcf_file: str) -> DataFrame:
    """
<<<<<<< HEAD
    Compute global ancestry proportions directly from the `POP` field.

    Counts all haplotypes across variants for each sample and normalizes
    to per-sample fractions. A chromosome label is added from the filename.
=======
    Calculate per-sample global ancestry proportions from a VCF file with GT:POP.

    Adds a 'chrom' column extracted from the filename (e.g., 'chr21.vcf.gz').
>>>>>>> 7db3c01 (added chromosome column and updated matching to handle chrX cases; we don't expect these, but good to have already)

    Parameters
    ----------
    vcf_file : str
<<<<<<< HEAD
        Path to ancestry-annotated VCF.
=======
        Path to the VCF file.
>>>>>>> 7db3c01 (added chromosome column and updated matching to handle chrX cases; we don't expect these, but good to have already)

    Returns
    -------
    DataFrame
        Global ancestry proportions with columns: sample_id, <ancestries...>, chrom
    """
    # Get information from VCF
    vcf = VCF(vcf_file)
    samples = vcf.samples
    ancestries = _parse_pop_labels(vcf_file)
    anc_index = {label: i for i, label in enumerate(ancestries)}
    n_samples = len(samples)
    n_ancestries = len(ancestries)

    # counts: sample x ancestry
    global_counts = zeros((n_samples, n_ancestries), dtype=int32)
    total_alleles = zeros(n_samples, dtype=int32)

    for rec in vcf:
        pop_fields = rec.format("POP")
        if pop_fields is None:
            continue
        
        for s_idx, pop_string in enumerate(pop_fields):
            if not pop_string:
                continue

            haps = [hap.strip() for hap in pop_string.split(",") if hap.strip()]
            for hap in haps:
                global_counts[s_idx, anc_index[hap]] += 1
                total_alleles[s_idx] += 1

    # Normalize to fractions
    fractions = global_counts / total_alleles[:, None]
    df = DataFrame(fractions, columns=ancestries)
    df.insert(0, "sample_id", samples)

    # Extract chromosome from filename (e.g., "chr21" or "chrX")
    m = search(r'chr[\w]+', vcf_file)
    if m:
        chrom = m.group(0)
        df["chrom"] = chrom
    else:
        print(f"Warning: Could not extract chromosome information from '{vcf_file}'")
        df["chrom"] = None

    return df


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
