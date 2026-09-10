import numpy as np
import pandas as pd
import pytest
import dask.array as da

from rfmix_reader.io._layout import flatten_names, sample_id_list, to_2d


def _g_anc(samples=("S1", "S2", "S3"), pops=("EUR", "AFR")):
    df = pd.DataFrame({"sample_id": list(samples), "chrom": ["chr1"] * len(samples)})
    for p in pops:
        df[p] = 0.5
    return df[["sample_id", *pops, "chrom"]]


def test_flatten_names_is_sample_major():
    assert flatten_names(["S1", "S2"], ["A", "B"]) == ["S1_A", "S1_B", "S2_A", "S2_B"]


def test_sample_id_list():
    assert sample_id_list(_g_anc()) == ["S1", "S2", "S3"]


def test_to_2d_matches_reshape_and_names():
    L, S, A = 4, 3, 2
    data = np.arange(L * S * A, dtype=np.int8).reshape(L, S, A)
    admix = da.from_array(data, chunks=(2, 2, 1))   # ancestry axis split on purpose
    flat, names = to_2d(admix, _g_anc())
    assert flat.shape == (L, S * A)
    np.testing.assert_array_equal(flat.compute(), data.reshape(L, -1))
    assert names == flatten_names(["S1", "S2", "S3"], ["EUR", "AFR"])
    # column s*A + a holds admix[:, s, a]
    for s in range(S):
        for a in range(A):
            np.testing.assert_array_equal(flat[:, s * A + a].compute(), data[:, s, a])
            assert names[s * A + a] == f"S{s + 1}_{['EUR', 'AFR'][a]}"


def test_to_2d_single_sample():
    data = np.arange(24, dtype=np.int8).reshape(4, 3, 2)
    sel, names = to_2d(da.from_array(data), _g_anc(), sample_idx=1)
    np.testing.assert_array_equal(sel.compute(), data[:, 1, :])
    assert names == ["S2_EUR", "S2_AFR"]


def test_to_2d_accepts_numpy_and_2d_passthrough():
    data = np.zeros((4, 3, 2), dtype=np.int8)
    flat, names = to_2d(data, _g_anc())
    assert flat.shape == (4, 6)
    flat2, names2 = to_2d(flat, _g_anc())
    assert flat2.shape == (4, 6) and names2 == names
    sel, sel_names = to_2d(flat, _g_anc(), sample_idx=2)
    assert sel.shape == (4, 2) and sel_names == ["S3_EUR", "S3_AFR"]


def test_to_2d_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        to_2d(da.zeros((4, 2, 2), dtype=np.int8), _g_anc())
    with pytest.raises(ValueError):
        to_2d(da.zeros((4, 5), dtype=np.int8), _g_anc())
    with pytest.raises(ValueError):
        to_2d(da.zeros((4,), dtype=np.int8), _g_anc())


def test_to_2d_bad_sample_idx():
    with pytest.raises(IndexError):
        to_2d(da.zeros((4, 3, 2), dtype=np.int8), _g_anc(), sample_idx=3)
