"""
Helpers shared by the parsers: population order / global-ancestry alignment,
chromosome labels, and output-file discovery.
"""
from __future__ import annotations

import gzip
import logging
import warnings
from os.path import basename, exists
from pathlib import Path
from re import search as rsearch
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "pops_by_code", "align_g_anc_columns", "check_pop_order_consistent", "read_rfmix_q",
    "normalize_chrom_label", "extract_chrom_from_path", "chrom_sort_key",
    "filter_file_maps_by_chrom", "filter_paths_by_chrom", "get_prefixes",
    "get_pops", "get_sample_names", "sample_id_list", "flatten_names",
]

logger = logging.getLogger(__name__)

_MODE_SUFFIXES: Dict[str, List[str]] = {
    "rfmix": ["fb.tsv", "fb.tsv.gz", "rfmix.Q", "rfmix.Q.gz"],
    "msp": ["msp.tsv", "msp.tsv.gz", "rfmix.Q", "rfmix.Q.gz"],
    "flare": ["anc.vcf.gz", "global.anc.gz"],
}
_ALL_SUFFIXES: List[str] = sorted(
    {sfx for sfxs in _MODE_SUFFIXES.values() for sfx in sfxs}, key=len, reverse=True
)


def pops_by_code(mapping: Mapping[str, int]) -> List[str]:
    """
    Return population labels ordered by their integer code.

    Raises ``ValueError`` if the codes are not exactly ``0 .. n-1``.
    """
    if not mapping:
        raise ValueError("Population mapping is empty.")
    codes = sorted(int(c) for c in mapping.values())
    if codes != list(range(len(mapping))):
        raise ValueError(
            f"Population codes must be exactly 0..{len(mapping) - 1}, got {codes}."
        )
    return [label for label, _ in sorted(mapping.items(), key=lambda kv: int(kv[1]))]


def align_g_anc_columns(g_anc, pops: Sequence[str]):
    """
    Reorder the ancestry columns of ``g_anc`` to ``pops``.

    The result has columns ``["sample_id", *pops, "chrom"]`` (``chrom`` only if
    present).  Raises ``ValueError`` if the ancestry column set differs.
    """
    meta = [c for c in ("sample_id", "chrom") if c in g_anc.columns]
    present = [c for c in g_anc.columns if c not in meta]
    if set(present) != set(pops):
        raise ValueError(
            "Global ancestry columns do not match the local ancestry populations: "
            f"g_anc has {sorted(present)}, local ancestry has {sorted(pops)}."
        )
    ordered = ["sample_id"] + list(pops) + (["chrom"] if "chrom" in g_anc.columns else [])
    return g_anc[ordered]


def check_pop_order_consistent(pop_lists: Iterable[Sequence[str]], kind: str = "dataset") -> List[str]:
    """
    Verify that every per-file population list is identical and return it.
    """
    pop_lists = [list(p) for p in pop_lists]
    if not pop_lists:
        raise ValueError(f"No {kind} files were read.")
    first = pop_lists[0]
    for other in pop_lists[1:]:
        if other != first:
            raise ValueError(
                f"Population order differs between {kind} files: {first} vs {other}. "
                "All files must list the same populations in the same order."
            )
    return first


def read_rfmix_q(fn: str, add_chrom: bool = True) -> pd.DataFrame:
    """
    Read an RFMix ``.rfmix.Q`` global-ancestry table.

    Format::

        #rfmix diploid global ancestry .Q format output
        #sample  AFR  EUR
        Sample_1 1.00000 0.00000

    Returns a DataFrame with ``sample_id``, one float32 column per ancestry in
    file order, and ``chrom`` (``"chr<label>"``) when ``add_chrom`` is true and
    the label can be inferred from the file name.
    """

    opener = gzip.open if str(fn).endswith(".gz") else open
    with opener(fn, "rt") as fh:
        first = fh.readline()
        second = fh.readline().strip()
    header_line = second if second.startswith("#") else first.strip()
    if not header_line.startswith("#"):
        raise ValueError(
            f"Could not parse Q header from '{fn}'. Expected a line starting "
            "with '#sample'."
        )
    header = header_line.lstrip("#").split()
    if not header or header[0].lower() != "sample":
        raise ValueError(f"Could not parse sample column from Q header in '{fn}'.")
    pops = header[1:]

    df = pd.read_csv(
        fn, sep=r"\s+", comment="#", header=None,
        names=["sample_id", *pops], compression="infer",
        dtype={p: np.float32 for p in pops},
    )
    df["sample_id"] = df["sample_id"].astype(str)
    if add_chrom:
        label = extract_chrom_from_path(str(fn))
        if label is not None:
            df["chrom"] = f"chr{label}"
        else:
            warnings.warn(
                f"Could not infer a chromosome label from '{fn}'; "
                "'chrom' column not added.", stacklevel=2,
            )
    return df


def normalize_chrom_label(label: str) -> str:
    """Normalize chromosome labels by stripping a ``chr`` prefix and lowering."""

    label = label.lower()
    return label[3:] if label.startswith("chr") else label


def extract_chrom_from_path(path: str) -> Optional[str]:
    """Best-effort extraction of a chromosome label from a file path."""

    base = basename(path).lower()
    match = rsearch(r"chr([a-z0-9]+)", base)
    if match:
        return match.group(1)

    fallback = rsearch(r"(?:[_\.])([0-9xy]+)(?:[^a-z0-9]|$)", base)
    if fallback:
        return fallback.group(1)

    return None


def filter_file_maps_by_chrom(
    file_maps: List[dict], chrom: Optional[str], *, kind: str = "dataset",
) -> List[dict]:
    """
    Filter file maps produced by :func:`get_prefixes` to a single chromosome.

    Parameters
    ----------
    file_maps
        List of dictionaries mapping suffixes to file paths.
    chrom
        Target chromosome label. When :data:`None`, the input is returned
        unchanged.
    kind
        Label used in error messages to clarify what is being filtered.
    """

    if chrom is None:
        return file_maps

    target = normalize_chrom_label(str(chrom))
    filtered: List[dict] = []

    for fmap in file_maps:
        paths = list(fmap.values())
        chrom_label = extract_chrom_from_path(paths[0]) if paths else None
        if chrom_label is None:
            continue
        if normalize_chrom_label(chrom_label) == target:
            filtered.append(fmap)

    if not filtered:
        raise FileNotFoundError(
            f"No {kind} files found for chromosome '{chrom}'."
        )

    return filtered


def filter_paths_by_chrom(
    paths: List[str], chrom: Optional[str], *, kind: str = "VCF"
) -> List[str]:
    """Filter a list of file paths down to those matching ``chrom``."""

    if chrom is None:
        return paths

    target = normalize_chrom_label(str(chrom))
    filtered: List[str] = []

    for path in paths:
        chrom_label = extract_chrom_from_path(path)
        if chrom_label is None:
            continue
        if normalize_chrom_label(chrom_label) == target:
            filtered.append(path)

    if not filtered:
        raise FileNotFoundError(
            f"No {kind} files found for chromosome '{chrom}'."
        )

    return filtered


def _suffix_key(sfx: str) -> str:
    """Normalised file-map key for a suffix (``fb.tsv.gz`` -> ``fb.tsv``)."""
    return sfx[:-3] if sfx.endswith(".gz") else sfx


def _strip_known_suffix(path: str, suffixes: Sequence[str]) -> Optional[str]:
    """Return ``path`` without its (longest) known suffix, or ``None``."""
    for sfx in sorted(suffixes, key=len, reverse=True):
        if path.endswith("." + sfx):
            return path[: -len(sfx) - 1]
    return None


def chrom_sort_key(prefix: str):
    """Sort prefixes by numeric chromosome first (chr2 before chr10), then name."""
    label = extract_chrom_from_path(prefix)
    if label is None:
        return (2, 0, prefix)
    if label.isdigit():
        return (0, int(label), prefix)
    return (1, 0, label + prefix)


def _clean_prefixes(prefixes: Sequence[str], suffixes: Optional[Sequence[str]] = None) -> List[str]:
    """
    Reduce a list of file paths to unique, sorted path prefixes.

    Each path is stripped of its (longest) known output suffix, e.g.
    ``/out/cohort.v2_chr1.fb.tsv.gz`` -> ``/out/cohort.v2_chr1``.  Paths that
    do not end in a known suffix (logs, indexes, ...) are dropped.

    Parameters
    ----------
    prefixes : sequence of str
        File paths.
    suffixes : sequence of str, optional
        Suffixes to recognise.  Default: every suffix of every mode.
    """
    suffixes = list(suffixes) if suffixes else _ALL_SUFFIXES
    cleaned = []
    for path in prefixes:
        stem = _strip_known_suffix(str(path), suffixes)
        if stem is not None:
            cleaned.append(stem)
    return sorted(dict.fromkeys(cleaned), key=chrom_sort_key)


def _discover_prefixes(file_prefix: str, suffixes: Sequence[str]) -> List[str]:
    """
    Find output-file prefixes for ``file_prefix``.

    ``file_prefix`` may be a directory (every file inside is considered), a
    complete file path, or a path prefix (``/out/run_`` matches
    ``/out/run_chr1.fb.tsv``, ``/out/run_chr2.fb.tsv``, ...).
    """
    p = Path(file_prefix)
    if p.is_dir():
        candidates = [str(x) for x in p.iterdir() if x.is_file()]
    elif p.is_file():
        candidates = [str(p)]
    else:
        candidates = [str(x) for x in p.parent.glob(p.name + "*") if x.is_file()]
    return _clean_prefixes(candidates, suffixes)


def _build_file_maps(prefixes: Sequence[str], suffixes: Sequence[str]) -> List[Dict[str, str]]:
    """Map each prefix to ``{normalised suffix: existing path}``; plain files win over ``.gz``."""
    fn = []
    for pfx in prefixes:
        filemap: Dict[str, str] = {}
        for sfx in suffixes:
            key = _suffix_key(sfx)
            if key in filemap:
                continue
            candidate = f"{pfx}.{sfx}"
            if exists(candidate):
                filemap[key] = candidate
        if filemap:
            fn.append(filemap)
    return fn


def get_prefixes(file_prefix: str, mode: str = "rfmix", verbose: bool = True) -> List[Dict[str, str]]:
    """
    Locate RFMix / FLARE output files and group them per chromosome.

    Parameters
    ----------
    file_prefix : str
        A directory containing the outputs, a single output file, or a common
        path prefix of the outputs (``"/out/run_"``).
    mode : {"rfmix", "msp", "flare"}
        - ``"rfmix"``: ``<prefix>.fb.tsv[.gz]`` and ``<prefix>.rfmix.Q[.gz]``
        - ``"msp"``:   ``<prefix>.msp.tsv[.gz]`` and ``<prefix>.rfmix.Q[.gz]``
        - ``"flare"``: ``<prefix>.anc.vcf.gz`` and ``<prefix>.global.anc.gz``
    verbose : bool, optional
        Print the order in which multiple file sets are read.

    Returns
    -------
    list of dict
        One dict per prefix mapping the normalised suffix (``"fb.tsv"``,
        ``"rfmix.Q"``, ``"msp.tsv"``, ``"anc.vcf"``, ``"global.anc"``) to the
        existing file path.  Only prefixes that have the primary file
        (first suffix of the mode) are returned.  Sorted by chromosome.

    Raises
    ------
    FileNotFoundError
        If no primary files are found.
    ValueError
        If ``mode`` is unknown.
    """
    if mode not in _MODE_SUFFIXES:
        raise ValueError(
            f"Invalid mode: {mode}. Choose from {list(_MODE_SUFFIXES.keys())}."
        )
    suffixes = _MODE_SUFFIXES[mode]
    primary = _suffix_key(suffixes[0])

    prefixes = _discover_prefixes(file_prefix, suffixes)
    fn = [m for m in _build_file_maps(prefixes, suffixes) if primary in m]
    if not fn:
        raise FileNotFoundError(
            f"No valid {mode.upper()} files found for prefix: {file_prefix}"
        )

    if len(fn) > 1 and verbose:
        names = [basename(m[primary]) for m in fn]
        logger.info("Multiple %s file sets read in this order: %s", mode.upper(), names)
    return fn


def get_pops(g_anc: pd.DataFrame):
    """
    Extract population names from an RFMix Q-matrix DataFrame.

    This function removes the 'sample_id' and 'chrom' columns from
    the input DataFrame and returns the remaining column names, which
    represent population names.

    Parameters
    ----------
    g_anc (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
        Expected to have 'sample_id' and 'chrom' columns, along with
        population columns.

    Returns
    -------
    np.ndarray: An array of population names extracted from the column names.

    Example
    -------
    If g_anc has columns ['sample_id', 'chrom', 'pop1', 'pop2', 'pop3'],
    this function will return ['pop1', 'pop2', 'pop3'].

    Note
    ----
    This function assumes that all columns other than 'sample_id' and 'chrom'
    represent population names.
    """
    return g_anc.drop(["sample_id", "chrom"], axis=1).columns.values


def get_sample_names(g_anc: pd.DataFrame):
    """
    Extract unique sample IDs from an RFMix Q-matrix DataFrame and
    convert to Arrow array.

    This function retrieves unique values from the 'sample_id' column
    of the input DataFrame and converts them to a PyArrow array.

    Parameters
    ----------
    g_anc (pd.DataFrame): A DataFrame containing RFMix Q-matrix data.
        Expected to have a 'sample_id' column.

    Returns
    -------
    pa.Array: A PyArrow array containing unique sample IDs.

    Example
    -------
    If g_anc has a 'sample_id' column with values ['sample1', 'sample2',
    'sample1', 'sample3'], this function will return a PyArrow array
    containing ['sample1', 'sample2', 'sample3'].

    Note
    ----
    This function assumes that the 'sample_id' column exists in the
    input DataFrame. It uses PyArrow on GPU for efficient memory
    management and interoperability with other data processing libraries.
    """
    if hasattr(g_anc, "to_pandas"):
        return g_anc.sample_id.unique().to_arrow()
    else:
        return g_anc.sample_id.unique()


def _as_list(values) -> List[str]:
    if hasattr(values, "to_pylist"):       # pyarrow array (cuDF path)
        values = values.to_pylist()
    elif hasattr(values, "to_pandas"):     # cuDF series
        values = values.to_pandas().tolist()
    return [str(v) for v in list(values)]


def sample_id_list(g_anc) -> List[str]:
    """Unique sample IDs of ``g_anc`` as a plain list of str."""
    return _as_list(get_sample_names(g_anc))


def flatten_names(sample_ids: Sequence[str], pops: Sequence[str]) -> List[str]:
    """Sample-major column names ``f"{sample}_{pop}"``."""
    return [f"{sample}_{pop}" for sample in sample_ids for pop in pops]
