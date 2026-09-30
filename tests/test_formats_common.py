"""Helpers in formats.common and core.codes: codes, population order, labels, discovery."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rfmix_reader.core.codes import MISSING, codes_from_counts, counts_from_hap_codes
from rfmix_reader.formats import common


# ------------------------------------------------------------------ codes
def test_counts_from_hap_codes_basic():
    out = counts_from_hap_codes(np.array([0, 1, -1, 2, 1]), np.array([1, 1, 0, 0, 255]), n_anc=2)
    assert out.dtype == np.int8 and MISSING == -1
    np.testing.assert_array_equal(out, [[1, 1], [0, 2], [-1, -1], [-1, -1], [-1, -1]])
    with pytest.raises(ValueError):
        counts_from_hap_codes(np.zeros(2), np.zeros(3), 2)


def test_codes_from_counts_round_trip():
    counts = np.array([[2, 0], [1, 1], [0, 2], [-1, -1], [1, 0]], dtype=np.int8)
    np.testing.assert_array_equal(codes_from_counts(counts), [[0, 0], [0, 1], [1, 1], [-1, -1], [-1, -1]])


# ------------------------------------------------------------------ populations
def test_pops_by_code_and_alignment():
    assert common.pops_by_code({"EUR": 0, "AFR": 1}) == ["EUR", "AFR"]
    assert common.pops_by_code({"AFR": 1, "NAT": 2, "EUR": 0}) == ["EUR", "AFR", "NAT"]
    with pytest.raises(ValueError):
        common.pops_by_code({"EUR": 0, "AFR": 2})
    g = pd.DataFrame({"sample_id": ["a"], "AFR": [0.3], "EUR": [0.7], "chrom": ["chr1"]})
    assert list(common.align_g_anc_columns(g, ["EUR", "AFR"]).columns) == ["sample_id", "EUR", "AFR", "chrom"]
    with pytest.raises(ValueError, match="do not match"):
        common.align_g_anc_columns(g, ["EUR", "NAT"])
    assert common.check_pop_order_consistent([["A", "B"], ["A", "B"]]) == ["A", "B"]
    with pytest.raises(ValueError, match="differs"):
        common.check_pop_order_consistent([["A", "B"], ["B", "A"]], kind="MSP")


def test_read_rfmix_q(msp_dir):
    df = common.read_rfmix_q(str(msp_dir / "chr1.rfmix.Q"))
    assert list(df.columns) == ["sample_id", "EUR", "AFR", "chrom"]
    assert df["chrom"].iloc[0] == "chr1" and df["EUR"].dtype == np.float32
    assert "chrom" not in common.read_rfmix_q(str(msp_dir / "chr1.rfmix.Q"), add_chrom=False).columns


def test_get_pops_and_sample_names():
    df = pd.DataFrame({"sample_id": ["S1", "S2"], "chrom": ["chr1"] * 2, "AFR": [0.1, 0.2], "EUR": [0.9, 0.8]})
    assert list(common.get_pops(df)) == ["AFR", "EUR"]
    assert common.sample_id_list(df) == ["S1", "S2"]
    assert common.flatten_names(["S1", "S2"], ["A", "B"]) == ["S1_A", "S1_B", "S2_A", "S2_B"]


# ------------------------------------------------------------------ chromosome labels
def test_chrom_label_helpers():
    assert common.normalize_chrom_label("ChrX") == "x"
    assert common.normalize_chrom_label("12") == "12"
    assert common.extract_chrom_from_path("/tmp/sample_chr10.fb.tsv") == "10"
    assert common.extract_chrom_from_path("/tmp/run_12.fb.tsv") == "12"
    assert common.extract_chrom_from_path("/tmp/misc.txt") is None
    names = ["chr10.fb.tsv", "chr2.fb.tsv", "chr1.fb.tsv", "chrX.fb.tsv", "misc"]
    assert sorted(names, key=common.chrom_sort_key) == ["chr1.fb.tsv", "chr2.fb.tsv", "chr10.fb.tsv",
                                                        "chrX.fb.tsv", "misc"]


def test_filter_by_chrom():
    maps = [{"fb.tsv": "/d/run_chr1.fb.tsv"}, {"fb.tsv": "/d/run_chr2.fb.tsv"}, {"fb.tsv": "/d/nochrom.fb.tsv"}]
    assert common.filter_file_maps_by_chrom(maps, "1", kind="t") == [maps[0]]
    assert common.filter_file_maps_by_chrom(maps, None) == maps
    with pytest.raises(FileNotFoundError):
        common.filter_file_maps_by_chrom(maps, "22", kind="t")
    assert common.filter_paths_by_chrom(["/d/chr1.x", "/d/chr2.x"], "chr2") == ["/d/chr2.x"]


# ------------------------------------------------------------------ discovery
def _touch(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_text("x")


def test_get_prefixes_directory_prefix_and_file(tmp_path):
    _touch(tmp_path, "run_chr1.fb.tsv", "run_chr1.rfmix.Q", "run_chr2.fb.tsv", "other_chr3.fb.tsv")
    out = common.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert [Path(m["fb.tsv"]).name for m in out] == ["run_chr1.fb.tsv", "run_chr2.fb.tsv", "other_chr3.fb.tsv"]
    out = common.get_prefixes(str(tmp_path / "run_"), mode="rfmix", verbose=False)
    assert [Path(m["fb.tsv"]).name for m in out] == ["run_chr1.fb.tsv", "run_chr2.fb.tsv"]
    assert "rfmix.Q" in out[0] and "rfmix.Q" not in out[1]
    one = common.get_prefixes(str(tmp_path / "run_chr1"), mode="rfmix", verbose=False)
    assert len(one) == 1 and set(one[0]) == {"fb.tsv", "rfmix.Q"}
    assert len(common.get_prefixes(str(tmp_path / "run_chr2.fb.tsv"), mode="rfmix", verbose=False)) == 1


def test_get_prefixes_dotted_names_gz_and_modes(tmp_path):
    _touch(tmp_path, "cohort.v2_chr1.fb.tsv", "xyz.fb.tsv", "abc.logs", "chr1.fb.tsv.tbi", "chr2.fb.tsv.gz")
    out = common.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert sorted(Path(m["fb.tsv"]).name for m in out) == ["chr2.fb.tsv.gz", "cohort.v2_chr1.fb.tsv", "xyz.fb.tsv"]
    _touch(tmp_path, "chr2.fb.tsv")
    out = common.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert "chr2.fb.tsv" in [Path(m["fb.tsv"]).name for m in out]      # plain preferred over gz
    with pytest.raises(FileNotFoundError):
        common.get_prefixes(str(tmp_path), mode="msp", verbose=False)
    with pytest.raises(FileNotFoundError):
        common.get_prefixes(str(tmp_path), mode="flare", verbose=False)
    with pytest.raises(ValueError):
        common.get_prefixes(str(tmp_path), mode="nope")
    assert common._clean_prefixes(["/x/cohort.v2_chr1.fb.tsv", "/x/cohort.v2_chr1.rfmix.Q"]) == ["/x/cohort.v2_chr1"]
