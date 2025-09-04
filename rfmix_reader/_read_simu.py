"""
Revision of `_read_flare.py` to work with data generated from
`haptools simgenotype` with population field flag (`--pop_field`).
"""
from tqdm import tqdm
from glob import glob
from cyvcf2 import VCF
from numpy import int32
from dask import delayed
from os.path import isdir, join, isfile
from typing import Callable, List, Tuple, Iterator
from dask.array import Array, concatenate, from_delayed, stack

from ._utils import set_gpu_environment

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
        Path to BGZF compressed ancestry VCF.
    chunk_size : int
        Variants per chunk to load for efficiency.
    verbose : bool, optional
        :const:`True` for progress information; :const:`False` otherwise.
        Default:`True`.

    Returns
    -------
    loci_df : :class:`DataFrame`
        Loci information for the FB data.
    g_anc : :class:`DataFrame`
        Per-sample global ancestry proportions per chromosome.
    local_array : :class:`dask.array.Array`
        Local ancestry per population stacked (variants, samples, ancestries).
        This is in alphabetical order of the populations. This matches RFMix.
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


def _read_file(fn: List[str], read_func: Callable, pbar=None) -> List:
    """
    Read data from multiple files using a provided read function.

    Parameters:
    ----------
    fn : List[str]
        A list of file paths to read.
    read_func : Callable
        A function to read data from each file.
    pbar : Optional
        A progress bar object to update during reading.

    Returns:
    -------
    List: A list containing the data read from each file.
    """
    data = [];
    for file_name in fn:
        data.append(read_func(file_name))
        if pbar:
            pbar.update(1)
    return data


def _read_loci_from_vcf(
        vcf_file: str, chunk_size: int32 = 1_000_000
) -> Iterator[DataFrame]:
    """
    Load chromosomal position (loci) information from a BGZF-compressed VCF file.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file.
    chunk_size : int, optional
        Number of records to process per chunk.

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
    """Read the unique ancestries from POP field."""
    vcf = VCF(vcf_file)
    for rec in vcf:
        for sample_field in rec.format("POP"):
            if sample_field:
                parts = sample_field.split(",")
                return sorted(set(parts))
    raise ValueError("No POP field found in VCF.")


def _load_haplotypes_from_pop(vcf_file: str, chunk_size: int = 10_000) -> Array:
    """Generate the Dask array with haplotypes."""
    vcf = VCF(vcf_file)
    samples = vcf.samples
    n_samples = len(samples)
    ancestries = _parse_pop_labels(vcf_file)
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
    return stack([stacked[:, :, anc_index[anc]]
                  for anc in sorted(anc_index)], axis=2)


def _calculate_global_ancestry_from_pop(vcf_file: str) -> DataFrame:
    """Calculate global ancestry from POP."""
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
    df.insert(0, "SAMPLE", samples)
    return df


def _get_vcf_files(vcf_path: str) -> List[str]:
    if isdir(vcf_path):
        vcf_files = sorted(glob(join(vcf_path, "*.vcf.gz")))
    elif isfile(vcf_path) and vcf_path.endswith(".vcf.gz"):
        vcf_files = [vcf_path]
    else:
        raise ValueError(f"Invalid input: {vcf_path} must be a .vcf.gz file or directory containing them.")

    if not vcf_files:
        raise FileNotFoundError(f"No VCF files found in path: {vcf_path}")
    return vcf_files
