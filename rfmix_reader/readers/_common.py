"""
Shared helpers for the readers.

Everything here is plain numpy / pandas so it can be imported without dask or
any optional dependency.  The three conventions enforced by this module are:

* ``local_array`` is ``int8`` with values ``0/1/2`` (diploid ancestry counts)
  and :data:`MISSING` (``-1``) for a sample/locus with no ancestry call.
* Axis 2 of ``local_array`` follows the population order defined by the tool
  that produced the file (RFMix reference-panel order, FLARE ``##ANCESTRY``
  codes, ...).
* The ancestry columns of ``g_anc`` are in that same order, so
  ``get_pops(g_anc)`` labels axis 2 correctly.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MISSING",
    "counts_from_hap_codes",
    "pops_by_code",
    "align_g_anc_columns",
    "check_pop_order_consistent",
    "maybe_to_backend_frames",
    "read_rfmix_q",
    "maybe_report_gpu",
]

#: Sentinel stored in ``local_array`` where a sample has no ancestry call.
MISSING = np.int8(-1)


def counts_from_hap_codes(hap0, hap1, n_anc: int) -> np.ndarray:
    """
    Combine two haplotype ancestry-code arrays into diploid ancestry counts.

    Parameters
    ----------
    hap0, hap1 : array-like of int, same shape ``(...)``
        Ancestry code of each haplotype (``0 .. n_anc-1``).  Any value outside
        that range (negative, 255, ...) marks a missing call.
    n_anc : int
        Number of ancestries.

    Returns
    -------
    np.ndarray, dtype int8, shape ``(..., n_anc)``
        Counts per ancestry.  Rows where either haplotype is missing are set
        to :data:`MISSING` for every ancestry.
    """
    hap0 = np.asarray(hap0)
    hap1 = np.asarray(hap1)
    if hap0.shape != hap1.shape:
        raise ValueError("hap0 and hap1 must have the same shape.")

    h0 = hap0.astype(np.int64, copy=False)
    h1 = hap1.astype(np.int64, copy=False)
    valid = (h0 >= 0) & (h0 < n_anc) & (h1 >= 0) & (h1 < n_anc)

    eye = np.eye(n_anc, dtype=np.int8)
    out = eye[np.clip(h0, 0, n_anc - 1)] + eye[np.clip(h1, 0, n_anc - 1)]
    if not valid.all():
        out[~valid] = MISSING
    return out


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


def maybe_to_backend_frames(*frames):
    """
    Convert pandas DataFrames to cuDF when cuDF is the selected backend.

    Readers always parse with pandas; this is applied once, on the returned
    frames, so GPU installs keep receiving cuDF objects without any GPU logic
    inside the parsers.
    """
    from ..backends import _select_dataframe_backend

    df_mod = _select_dataframe_backend()
    if df_mod.__name__ != "cudf":
        return frames if len(frames) != 1 else frames[0]

    converted = []
    for df in frames:
        if df is None or not isinstance(df, pd.DataFrame):
            converted.append(df)
        else:
            converted.append(df_mod.DataFrame.from_pandas(df))
    return tuple(converted) if len(converted) != 1 else converted[0]


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
    import gzip
    import warnings

    from ..utils import _extract_chrom_from_path

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
        label = _extract_chrom_from_path(str(fn))
        if label is not None:
            df["chrom"] = f"chr{label}"
        else:
            warnings.warn(
                f"Could not infer a chromosome label from '{fn}'; "
                "'chrom' column not added.", stacklevel=2,
            )
    return df


def maybe_report_gpu(verbose: bool) -> None:
    """Print GPU properties once when running with a cuDF backend and verbose."""
    if not verbose:
        return
    from ..backends import _select_dataframe_backend

    if _select_dataframe_backend().__name__ == "cudf":
        from ..utils import set_gpu_environment

        set_gpu_environment()
