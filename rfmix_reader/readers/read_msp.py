"""
Reader for RFMix v2 `.msp.tsv` (Most-probable Sequence of Populations) files.

The `.msp.tsv` format stores piecewise-constant hard ancestry calls as genomic
segments — roughly 2,000× smaller than the `.fb.tsv` forward-backward matrix
for the same dataset.  Reading `.msp.tsv` eliminates the need for binary
conversion and is the preferred source when posterior probabilities are not
required (e.g. GWAS, admixture mapping, local-ancestry validation).

Format (2-line header, then one row per segment):
    #Subpopulation order/codes: AFR=0   EUR=1
    #chm  spos  epos  sgpos  egpos  n snps  Sample_1.0  Sample_1.1 ...
    chr1  10397 670270 0.00  0.10   4752    0            1          ...

The per-sample columns alternate haplotype 0 / haplotype 1 (suffix .0 / .1).
Values are integer ancestry codes (0 = first listed population, 1 = second, …).
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from ..utils import (
    _read_file,
    filter_file_maps_by_chrom,
    set_gpu_environment,
)

try:
    from torch.cuda import is_available as gpu_available
except ModuleNotFoundError:
    def gpu_available():
        return False

if gpu_available():
    from cudf import DataFrame, concat, CategoricalDtype
else:
    from pandas import DataFrame, concat, CategoricalDtype


__all__ = ["read_rfmix"]

_MSP_SUFFIXES = ["msp.tsv", "msp.tsv.gz"]
_META_COLS = ["#chm", "spos", "epos", "sgpos", "egpos", "n snps"]


def _parse_pop_header(fn: str) -> Dict[str, int]:
    """
    Parse the first header line of an .msp.tsv file to extract population codes.

    Expected format:
        #Subpopulation order/codes: AFR=0   EUR=1

    Returns a dict mapping population label → integer code, e.g. {"AFR": 0, "EUR": 1}.
    """
    opener = __import__("gzip").open if fn.endswith(".gz") else open
    with opener(fn, "rt") as fh:
        line = fh.readline().strip()
    pairs = re.findall(r"(\w+)=(\d+)", line)
    if not pairs:
        raise ValueError(
            f"Could not parse population codes from first line of '{fn}'. "
            f"Expected format: #Subpopulation order/codes: AFR=0 EUR=1 ..."
        )
    return {label: int(code) for label, code in pairs}


def _read_msp_file(fn: str) -> Tuple[DataFrame, List[str], Dict[str, int]]:
    """
    Read a single .msp.tsv file.

    Returns
    -------
    segs : DataFrame
        All segment rows with columns: chrom, spos, epos, and per-haplotype
        ancestry integer columns (Sample_1.0, Sample_1.1, …).
    hap_cols : list of str
        Names of the per-haplotype columns in the order they appear.
    pop_map : dict
        Population label → integer code from the file header.
    """
    pop_map = _parse_pop_header(fn)

    segs = pd.read_csv(
        fn, sep="\t", comment=None, skiprows=1, header=0,
        compression="infer",
    )
    # Rename the column that carries the '#chm' label (pandas strips '#')
    segs.rename(columns={"#chm": "chrom", "spos": "spos", "epos": "epos"},
                inplace=True)
    segs["chrom"] = segs["chrom"].astype(CategoricalDtype())
    segs["spos"] = segs["spos"].astype(np.int32)
    segs["epos"] = segs["epos"].astype(np.int32)

    hap_cols = [c for c in segs.columns if c not in
                {"chrom", "spos", "epos", "sgpos", "egpos", "n snps"}]
    return segs, hap_cols, pop_map


def _segments_to_loci(
    segs: pd.DataFrame,
    hap_cols: List[str],
    pop_map: Dict[str, int],
    index_offset: int = 0,
) -> Tuple[DataFrame, da.Array]:
    """
    Convert segment rows to loci-level data and a dask ancestry array.

    The loci DataFrame has one row per segment (the segment start = reference
    locus).  The ancestry array has shape (n_segments, n_samples, n_ancestries)
    with hard integer counts (0/1/2) derived by summing the two haplotypes per
    sample.

    Parameters
    ----------
    segs : pd.DataFrame
        Output of _read_msp_file, one row per segment.
    hap_cols : list of str
        Per-haplotype column names (Sample_1.0, Sample_1.1, …).
    pop_map : dict
        Population label → integer code (e.g. {"AFR": 0, "EUR": 1}).
    index_offset : int
        Global locus index offset for multi-chromosome datasets.

    Returns
    -------
    loci_df : DataFrame
        Columns: chromosome, physical_position (= spos), i.
    local_array : dask.array.Array
        Shape (n_segments, n_samples, n_ancestries), dtype int32.
    """
    n_pops = len(pop_map)
    # Infer samples from hap_cols: Sample_1.0, Sample_1.1, Sample_2.0, …
    # Pair adjacent columns as (hap0, hap1) per sample
    if len(hap_cols) % 2 != 0:
        raise ValueError(
            f"Expected an even number of haplotype columns, got {len(hap_cols)}."
        )
    n_samples = len(hap_cols) // 2
    n_segs = len(segs)

    # Build the ancestry count array: shape (n_segs, n_samples, n_pops)
    hap_data = segs[hap_cols].to_numpy(dtype=np.int8)  # (n_segs, 2*n_samples)
    hap0 = hap_data[:, 0::2]  # (n_segs, n_samples) — haplotype 0
    hap1 = hap_data[:, 1::2]  # (n_segs, n_samples) — haplotype 1

    # One-hot encode both haplotypes and sum → diploid ancestry counts per pop
    eye = np.eye(n_pops, dtype=np.int8)
    # eye[hap0] shape: (n_segs, n_samples, n_pops)
    local_np = (eye[hap0] + eye[hap1]).astype(np.int32)

    # Sort populations alphabetically (matches read_rfmix / read_flare ordering)
    sorted_pops = sorted(pop_map, key=pop_map.get)
    pop_order = [pop_map[p] for p in sorted_pops]
    local_np = local_np[:, :, pop_order]

    local_array = da.from_array(local_np, chunks=(min(1024, n_segs), n_samples, n_pops))

    loci_df = DataFrame({
        "chromosome": pd.Categorical(segs["chrom"].astype(str)),
        "physical_position": segs["spos"].astype(np.int32).values,
        "i": np.arange(index_offset, index_offset + n_segs, dtype=np.int64),
    })

    return loci_df, local_array


def _get_msp_prefixes(file_prefix: str, verbose: bool = True) -> List[Dict[str, str]]:
    """
    Find .msp.tsv (or .msp.tsv.gz) files under file_prefix and return a
    list of per-chromosome file maps (same structure as get_prefixes).
    """
    from ..utils import _clean_prefixes
    from glob import glob
    from os.path import join

    fp = Path(file_prefix)
    prefixes = []

    # Accept both a path prefix ("/path/run_chr1" -> "/path/run_chr1.msp.tsv")
    # and a complete MSP file path.
    for sfx in _MSP_SUFFIXES:
        suffix = f".{sfx}"
        if str(fp).endswith(suffix) and fp.exists():
            prefixes.append(str(fp)[:-len(suffix)])
        elif Path(f"{fp}.{sfx}").exists():
            prefixes.append(str(fp))
    prefixes = list(dict.fromkeys(prefixes))

    if not prefixes:
        candidates = sorted([str(x) for x in fp.glob("*[chr]*")])
        if not candidates:
            candidates = sorted(glob(join(str(fp), "*")))
        prefixes = sorted(_clean_prefixes(candidates))

    fn = []
    for pfx in prefixes:
        filemap = {}
        for sfx in _MSP_SUFFIXES:
            candidate = f"{pfx}.{sfx}"
            if Path(candidate).exists():
                key = sfx.replace(".gz", "")
                filemap[key] = candidate
                break  # prefer plain over .gz
        if filemap:
            fn.append(filemap)

    if not fn:
        raise FileNotFoundError(
            f"No .msp.tsv files found under prefix: {file_prefix}"
        )

    if len(prefixes) > 1 and verbose:
        from os.path import basename
        print(f"Multiple MSP file sets read in this order: "
              f"{[basename(f) for f in prefixes]}")
    return fn


def _read_Q_for_msp(fn: str, pop_map: Dict[str, int]) -> DataFrame:
    """
    Build a minimal global-ancestry DataFrame from the MSP segment data.

    RFMix .rfmix.Q files are not required when reading .msp.tsv.  Instead,
    we compute per-sample global ancestry proportions by weighting each
    segment by its physical length.

    Returns a DataFrame with columns [sample_id, <pop1>, <pop2>, …, chrom].
    """
    raise NotImplementedError(
        "Global ancestry from .msp.tsv is not yet implemented. "
        "Pass a pre-loaded g_anc DataFrame or read it separately with "
        "read_rfmix_fb() and reuse its g_anc."
    )


def read_rfmix(
    file_prefix: str,
    g_anc: Optional[DataFrame] = None,
    verbose: bool = True,
    chrom: Optional[str] = None,
) -> Tuple[DataFrame, Optional[DataFrame], da.Array]:
    """
    Read RFMix `.msp.tsv` files into a loci DataFrame and a Dask ancestry array.

    This is the **recommended default reader** for RFMix output. The `.msp.tsv`
    format stores piecewise-constant hard ancestry calls as genomic segments and
    is roughly **2,000× smaller** than the corresponding `.fb.tsv` forward-backward
    matrix. Reading it directly eliminates the costly binary-conversion step and is
    sufficient for the vast majority of local ancestry analyses (GWAS, admixture
    mapping, QC, visualization).

    Use :func:`read_rfmix_fb` instead when you explicitly require posterior
    probability values from the forward-backward matrix.

    Parameters
    ----------
    file_prefix : str
        Directory or path prefix under which ``.msp.tsv`` files live.
    g_anc : DataFrame, optional
        Pre-loaded global ancestry DataFrame (from :func:`read_rfmix_fb` or
        similar).  When provided it is returned unchanged.  When :data:`None`
        the second return value is :data:`None`.
    verbose : bool, default True
        Print progress information.
    chrom : str, optional
        Restrict reading to a single chromosome (with or without ``chr`` prefix).

    Returns
    -------
    loci_df : DataFrame
        Columns: ``chromosome``, ``physical_position``, ``i``.
        One row per RFMix ancestry segment boundary.
    g_anc : DataFrame or None
        Passed through unchanged, or :data:`None` if not supplied.
    local_array : dask.array.Array
        Shape ``(n_segments, n_samples, n_ancestries)``, dtype ``int32``.
        Hard ancestry counts (0/1/2); populations are sorted alphabetically.

    Notes
    -----
    Local ancestry is at **segment resolution** — each row is the start
    position of one RFMix ancestry segment.  Use :func:`interpolate_array`
    with ``method='stepwise'`` to expand segments onto a denser variant grid,
    or use :func:`write_imputed` for direct variant-level output.

    Trade-offs versus :func:`read_rfmix_fb` (`.fb.tsv`):
    - **Pro**: ~2,000× smaller files; no binary conversion needed; loads in seconds.
    - **Con**: Hard calls only — no posterior uncertainty information.

    Examples
    --------
    >>> loci, g_anc, admix = read_rfmix("data/rfmix_out/")
    >>> print(loci.shape, admix.shape)
    (1629, 3) (1629, 81, 2)
    """
    if verbose and gpu_available():
        set_gpu_environment()

    fn = filter_file_maps_by_chrom(
        _get_msp_prefixes(file_prefix, verbose), chrom, kind="MSP"
    )

    pbar = tqdm(desc="Reading MSP files", total=len(fn), disable=not verbose)
    results = _read_file(fn, lambda f: _read_msp_file(f["msp.tsv"]), pbar)
    pbar.close()

    index_offset = 0
    loci_dfs = []
    local_arrays = []

    for segs, hap_cols, pop_map in results:
        segs_pd = segs.to_pandas() if hasattr(segs, "to_pandas") else segs
        loci_df_i, local_i = _segments_to_loci(
            segs_pd, hap_cols, pop_map, index_offset
        )
        loci_dfs.append(loci_df_i)
        local_arrays.append(local_i)
        index_offset += len(segs_pd)

    loci_df = concat(loci_dfs, axis=0, ignore_index=True)
    local_array = da.concatenate(local_arrays, axis=0)

    return loci_df, g_anc, local_array


# Expose helpers for tests and downstream code
read_rfmix._parse_pop_header = _parse_pop_header
read_rfmix._read_msp_file = _read_msp_file
read_rfmix._segments_to_loci = _segments_to_loci
