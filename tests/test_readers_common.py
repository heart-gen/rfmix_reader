import numpy as np
import pandas as pd
import pytest

from rfmix_reader.readers._common import (
    MISSING,
    align_g_anc_columns,
    check_pop_order_consistent,
    counts_from_hap_codes,
    pops_by_code,
)


def test_counts_from_hap_codes_basic():
    hap0 = np.array([0, 1, -1, 2, 1], dtype=np.int32)
    hap1 = np.array([1, 1, 0, 0, 255], dtype=np.int32)
    out = counts_from_hap_codes(hap0, hap1, n_anc=2)
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, [[1, 1], [0, 2], [-1, -1], [-1, -1], [-1, -1]])
    assert MISSING == -1


def test_counts_from_hap_codes_preserves_leading_shape():
    hap0 = np.zeros((3, 4), dtype=np.int8)
    hap1 = np.ones((3, 4), dtype=np.int8)
    out = counts_from_hap_codes(hap0, hap1, n_anc=3)
    assert out.shape == (3, 4, 3)
    np.testing.assert_array_equal(out[0, 0], [1, 1, 0])


def test_counts_from_hap_codes_shape_mismatch():
    with pytest.raises(ValueError):
        counts_from_hap_codes(np.zeros(2), np.zeros(3), 2)


def test_pops_by_code_orders_by_code():
    assert pops_by_code({"EUR": 0, "AFR": 1}) == ["EUR", "AFR"]
    assert pops_by_code({"AFR": 1, "NAT": 2, "EUR": 0}) == ["EUR", "AFR", "NAT"]


def test_pops_by_code_rejects_gaps():
    with pytest.raises(ValueError):
        pops_by_code({"EUR": 0, "AFR": 2})
    with pytest.raises(ValueError):
        pops_by_code({})


def test_align_g_anc_columns_reorders():
    g = pd.DataFrame({"sample_id": ["a"], "AFR": [0.3], "EUR": [0.7], "chrom": ["chr1"]})
    out = align_g_anc_columns(g, ["EUR", "AFR"])
    assert list(out.columns) == ["sample_id", "EUR", "AFR", "chrom"]


def test_align_g_anc_columns_without_chrom():
    g = pd.DataFrame({"sample_id": ["a"], "AFR": [0.3], "EUR": [0.7]})
    out = align_g_anc_columns(g, ["EUR", "AFR"])
    assert list(out.columns) == ["sample_id", "EUR", "AFR"]


def test_align_g_anc_columns_set_mismatch():
    g = pd.DataFrame({"sample_id": ["a"], "AFR": [0.3], "EUR": [0.7]})
    with pytest.raises(ValueError, match="do not match"):
        align_g_anc_columns(g, ["EUR", "NAT"])


def test_check_pop_order_consistent():
    assert check_pop_order_consistent([["A", "B"], ["A", "B"]]) == ["A", "B"]
    with pytest.raises(ValueError, match="differs"):
        check_pop_order_consistent([["A", "B"], ["B", "A"]], kind="MSP")
    with pytest.raises(ValueError):
        check_pop_order_consistent([])
