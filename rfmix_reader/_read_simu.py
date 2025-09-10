"""
Revision of `_read_flare.py` to work with data generated from
`haptools simgenotype` with population field flag (`--pop_field`).
"""
from re import sub
from tqdm import tqdm
from glob import glob
from cyvcf2 import VCF
from typing import List, Tuple, Iterator
from dask import delayed, compute as dask_compute
from pandas import DataFrame, concat, Categorical
from dask.array import Array, concatenate, from_delayed, stack
from os.path import isdir, join, isfile, dirname, basename, exists
from numpy import (
    int8, int64, int32,
    zeros, array, nan_to_num,
    eye, empty, asarray, frompyfunc
)

from ._utils import _read_file

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
    samples: List[str] = vcf.samples
    n_samples = len(samples)
    chrom = vcf.seqnames[0]

    # Extract ancestry labels
    ancestries = sorted(set(_parse_pop_labels(vcf_file)))
    n_ancestries = len(ancestries)
    I = eye(n_ancestries, dtype=int8) # One-hot encoding

    # Init global counters
    global_counts = zeros((n_samples, n_ancestries), dtype=int64)
    total_alleles = 0

    delayed_chunks = []
    batch = []
    batch_len = 0

    def _finalize_batch(batch_recs):
        """
        Turn a list[cyvcf2.Record] into one Dask chunk (variants, samples, ancestries), int8
        using fully vectorized operations.
        Also return global_counts increment for the batch.
        """
        n_vars = len(batch_recs)
        if n_vars == 0:
            return (zeros((0, n_samples, n_ancestries), dtype=int8),
                    zeros((n_samples, n_ancestries), dtype=int64),
                    0)

        pop_flat = empty(n_vars * n_samples, dtype=object)
        idx = 0
        for rec in batch_recs:
            pop = rec.format("POP") # array-like len = n_samples (strings "A,B")
            if pop is None:
                pop_flat[idx:idx+n_samples] = ""
                idx += n_samples
                continue
            pf = asarray(pop, dtype=object)
            # Fast-path decode if bytes
            mask_bytes = frompyfunc(lambda x: isinstance(x, (bytes, bytearray)), 1, 1)(pf).astype(bool)
            if mask_bytes.any():
                pf[mask_bytes] = [x.decode("utf-8", "ignore") for x in pf[mask_bytes]]
            pop_flat[idx:idx+n_samples] = pf
            idx += n_samples

        pop_mat = pop_flat.reshape(n_vars, n_samples)

        # Vectorize
        pop_u = pop_mat.astype("U16", copy=False)
        parts = np.char.partition(pop_u, ",") # shape (n_vars, n_samples, 3)
        h0 = parts[:, :, 0]
        h1 = parts[:, :, 2]

        # Integer-encode ancestry labels with C-level factorization
        h0_codes = Categorical(h0.ravel(), categories=ancestries,
                               ordered=False).codes.reshape(n_vars, n_samples)
        h1_codes = Categorical(h1.ravel(), categories=ancestries,
                               ordered=False).codes.reshape(n_vars, n_samples)

        # Build one-hot via table lookup
        valid0 = h0_codes >= 0
        valid1 = h1_codes >= 0

        # Build chunk
        local_chunk = zeros((n_vars, n_samples, n_ancestries), dtype=int8)

        if valid0.any():
            tmp0 = I.take(h0_codes, mode="clip", axis=0)
            tmp0[~valid0] = 0
            local_chunk += tmp0

        if valid1.any():
            tmp1 = I.take(h1_codes, mode="clip", axis=0)
            tmp1[~valid1] = 0
            local_chunk += tmp1

        # Global counts: sum over variants for each sample x ancestry
        batch_global = local_chunk.sum(axis=0, dtype=xp_int64)
        batch_alleles = 2 * n_vars

        return local_chunk, batch_global, batch_alleles

    while True:
        try:
            rec = next(vcf)
        except StopIteration:
            break
        batch.append(rec)
        batch_len += 1
        if batch_len >= chunk_size:
            # Close over a local copy to avoid late-binding issues
            records = batch
            batch = []
            batch_len = 0

            def _make_delayed(records_local):
                return delayed(_finalize_batch)(records_local)

            d = _make_delayed(records)
            # Split dask
            local_d = delayed(lambda x: x[0])(d)
            g_d     = delayed(lambda x: (x[1], x[2]))(d)

            # Collect number of variants
            n_vars_local = len(records)
            delayed_chunks.append((
                from_delayed(local_d, shape=(n_vars_local, n_samples, n_ancestries), dtype=int8),
                g_d
            ))

    # Flush remainder
    if batch_len > 0:
        records = batch
        d = delayed(_finalize_batch)(records)
        local_d = delayed(lambda x: x[0])(d)
        g_d     = delayed(lambda x: (x[1], x[2]))(d)
        delayed_chunks.append((
            from_delayed(local_d, shape=(len(records), n_samples, n_ancestries), dtype=int8),
            g_d
        ))

    # Separate local dask arrays and global contributions
    local_arrays = [lc for (lc, _) in delayed_chunks]
    global_pairs = [gd for (_, gd) in delayed_chunks]

    # Build one concatenated Dask array lazily
    local_array = concatenate(local_arrays, axis=0)

    # Aggregate global fractions
    global_contribs = dask_compute(*global_pairs)

    for g_inc, alleles_inc in global_contribs:
        global_counts += g_inc
        total_alleles += alleles_inc

    # Normalize global ancestry
    fractions = global_counts / total_alleles[:, None]
    fractions = nan_to_num(fractions, nan=0.0)

    # Build dataframe
    global_df = DataFrame(fractions, columns=ancestries)
    global_df.insert(0, "sample_id", samples)
    global_df["chrom"] = chrom

    return local_array, global_df


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
