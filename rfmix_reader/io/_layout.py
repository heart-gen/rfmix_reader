"""
Adapter between the readers' 3-D ``(loci, samples, ancestries)`` local
ancestry array and the 2-D ``(loci, samples * ancestries)`` layout consumed by
:func:`write_data` and :func:`admix_to_bed_individual`.

Column order is **sample-major**: ``S1_A, S1_B, S2_A, S2_B, ...`` — exactly the
order produced by ``admix.reshape(n_loci, -1)`` — and column names are built
with :func:`flatten_names` so labels always match the data.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from ..utils import get_pops, get_sample_names

if TYPE_CHECKING:
    from dask.array import Array

__all__ = ["flatten_names", "sample_id_list", "to_2d"]


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


def to_2d(admix, g_anc, sample_idx: Optional[int] = None) -> Tuple["Array", List[str]]:
    """
    Return ``admix`` as a 2-D dask array plus matching column names.

    Parameters
    ----------
    admix : dask.array.Array or numpy.ndarray
        ``(loci, samples, ancestries)`` from any reader, or an already 2-D
        ``(loci, samples * ancestries)`` array in sample-major order.
    g_anc : DataFrame
        Global ancestry table; supplies sample IDs and population labels
        (``get_sample_names`` / ``get_pops``).
    sample_idx : int, optional
        Select a single sample; the result is ``(loci, ancestries)`` with
        names for that sample only.

    Returns
    -------
    (array, names)
        Lazy 2-D dask array and the column names for its columns.

    Raises
    ------
    ValueError
        If the array shape does not match the sample / population counts of
        ``g_anc``.
    IndexError
        If ``sample_idx`` is out of range.
    """
    import dask.array as da

    pops = _as_list(get_pops(g_anc))
    sample_ids = sample_id_list(g_anc)
    n_samples, n_pops = len(sample_ids), len(pops)

    if isinstance(admix, np.ndarray):
        admix = da.from_array(admix, chunks="auto")

    if admix.ndim == 2:
        if admix.shape[1] != n_samples * n_pops:
            raise ValueError(
                f"2-D admix has {admix.shape[1]} columns but g_anc implies "
                f"{n_samples} samples x {n_pops} populations = {n_samples * n_pops}."
            )
        if sample_idx is not None:
            _check_sample_idx(sample_idx, n_samples)
            cols = slice(sample_idx * n_pops, (sample_idx + 1) * n_pops)
            return admix[:, cols], flatten_names([sample_ids[sample_idx]], pops)
        return admix, flatten_names(sample_ids, pops)

    if admix.ndim != 3:
        raise ValueError(f"admix must be 2-D or 3-D, got shape {admix.shape}.")
    if admix.shape[1] != n_samples or admix.shape[2] != n_pops:
        raise ValueError(
            f"admix shape {admix.shape} does not match g_anc: expected "
            f"(loci, {n_samples} samples, {n_pops} populations {pops})."
        )

    if sample_idx is not None:
        _check_sample_idx(sample_idx, n_samples)
        return admix[:, sample_idx, :], flatten_names([sample_ids[sample_idx]], pops)

    # Merge (samples, ancestries) -> one axis.  The ancestry axis must be a
    # single chunk for dask to merge it without a copy.
    if len(admix.chunks[2]) != 1:
        admix = admix.rechunk({2: n_pops})
    flat = admix.reshape(admix.shape[0], n_samples * n_pops)
    return flat, flatten_names(sample_ids, pops)


def _check_sample_idx(sample_idx: int, n_samples: int) -> None:
    if sample_idx < 0 or sample_idx >= n_samples:
        raise IndexError(
            f"sample_num {sample_idx} is out of range [0, {n_samples - 1}]"
        )
