"""
Haplotype ancestry codes and diploid counts.

* ``haplotype codes``: ``(..., 2)`` int8, one ancestry index per haplotype,
  :data:`MISSING` (``-1``) where there is no call.
* ``counts``: ``(..., n_ancestries)`` int8 diploid counts ``0/1/2``; a row is
  all :data:`MISSING` when either haplotype is missing.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MISSING", "counts_from_hap_codes", "codes_from_counts", "to_str_array"]

#: Sentinel for a sample/locus without an ancestry call.
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

    # Stay in the input dtype (int8 from the cache): no int64 upcast, no clip
    # copies.  Per 10k x 500 block this is ~30 MB of temporaries instead of
    # ~250 MB, which matters because dask keeps one block per thread in flight.
    valid = (hap0 >= 0) & (hap0 < n_anc) & (hap1 >= 0) & (hap1 < n_anc)
    i0 = np.where(valid, hap0, 0)
    i1 = np.where(valid, hap1, 0)

    eye = np.eye(n_anc, dtype=np.int8)
    out = eye[i0]
    out += eye[i1]
    if not valid.all():
        out[~valid] = MISSING
    return out


def codes_from_counts(counts) -> np.ndarray:
    """
    ``(..., A)`` diploid counts (0/1/2, ``-1`` missing) -> ``(..., 2)`` int8
    haplotype codes.  The phase is unknown, so the lower ancestry index is
    assigned to haplotype 0 (deterministic).  Float inputs are rounded.
    """
    c = np.asarray(counts)
    if c.dtype.kind == "f":
        c = np.where(np.isnan(c), -1, np.rint(c)).astype(np.int16)
    else:
        c = c.astype(np.int16, copy=False)
    missing = (c < 0).any(axis=-1)
    c = np.clip(c, 0, 2)
    cum = np.cumsum(c, axis=-1)
    hap0 = np.argmax(cum >= 1, axis=-1)
    hap1 = np.argmax(cum >= 2, axis=-1)
    total = c.sum(axis=-1)
    codes = np.stack([hap0, hap1], axis=-1).astype(np.int8)
    bad = missing | (total != 2)
    if bad.any():
        codes[bad] = MISSING
    return codes


def to_str_array(values) -> np.ndarray:
    """
    ``values`` as a 1-D object array of Python ``str``.

    Works for lists, fixed-width ``<U`` arrays, object arrays and numpy 2's
    variable-width ``StringDType`` (what Zarr returns for string coordinates),
    which ``.astype(str)`` cannot convert.
    """
    arr = np.asarray(values)
    if arr.dtype == object:
        return np.array([str(v) for v in arr.tolist()], dtype=object)
    return np.array([str(v) for v in arr.tolist()], dtype=object)
