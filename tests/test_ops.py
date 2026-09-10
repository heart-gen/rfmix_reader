"""Dataset operations: BED, positions, Parquet, interpolation, TAGORE."""
import numpy as np
import pandas as pd
import pytest

from rfmix_reader import open_rfmix
from rfmix_reader.core.legacy import codes_from_counts, from_legacy
from rfmix_reader.ops.bed import run_boundaries, to_bed
from rfmix_reader.ops.interpolate import build_variant_grid, interpolate
from rfmix_reader.ops.parquet import to_parquet
from rfmix_reader.ops.positions import at_positions


# ------------------------------------------------------------------ helpers
def test_codes_from_counts_round_trip():
    counts = np.array([[2, 0], [1, 1], [0, 2], [-1, -1], [1, 0]], dtype=np.int8)
    codes = codes_from_counts(counts)
    np.testing.assert_array_equal(codes, [[0, 0], [0, 1], [1, 1], [-1, -1], [-1, -1]])
    floats = np.array([[1.9, 0.1], [np.nan, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(codes_from_counts(floats), [[0, 0], [-1, -1]])
    three = np.array([[[0, 1, 1]]], dtype=np.int8)
    np.testing.assert_array_equal(codes_from_counts(three), [[[1, 2]]])


def test_from_legacy_matches_open(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    loci, g_anc, admix = ds.la.to_legacy()
    ds2 = from_legacy(loci, g_anc, admix)
    assert ds2.la.samples == ds.la.samples and ds2.la.ancestries == ds.la.ancestries
    np.testing.assert_array_equal(ds2.la.counts.values, ds.la.counts.values)
    assert ds2.la.chromosomes == ["chr1", "chr2"] and ds2.sizes["contig"] == 2
    pd.testing.assert_frame_equal(ds2.la.global_ancestry, ds.la.global_ancestry)


def test_from_legacy_without_g_anc():
    counts = np.zeros((3, 2, 2), dtype=np.int8) + np.array([1, 1], dtype=np.int8)
    loci = pd.DataFrame({"chromosome": ["chr1"] * 3, "physical_position": [1, 2, 3]})
    ds = from_legacy(loci, None, counts)
    assert ds.la.samples == ["Sample_1", "Sample_2"] and ds.la.ancestries == ["anc_0", "anc_1"]
    np.testing.assert_allclose(ds.la.global_ancestry[["anc_0", "anc_1"]].to_numpy(), 0.5)


# ------------------------------------------------------------------ BED
def test_run_boundaries():
    states = np.array([[2, 0], [2, 0], [1, 1], [0, 2], [0, 2], [2, 0]])
    assert run_boundaries(states) == [(0, 2, 0), (2, 3, 2), (3, 5, 3), (5, 6, 5)]
    # single-row runs (index 2 and the trailing index 5) are absorbed into the previous run
    assert run_boundaries(states, min_segment=2) == [(0, 3, 0), (3, 6, 3)]
    # a short leading run is absorbed into the following run
    assert run_boundaries(np.array([[1, 1], [2, 0], [2, 0]]), min_segment=2) == [(0, 3, 1)]
    assert run_boundaries(np.zeros((0, 2))) == []


def test_to_bed_from_reader(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    bed = to_bed(ds, "Sample_1")
    assert list(bed.columns) == ["chromosome", "start", "end", "Sample_1_EUR", "Sample_1_AFR"]
    chr1 = bed[bed["chromosome"] == "chr1"]
    # Sample_1 on chr1: EUR/EUR, EUR/AFR, AFR/AFR, AFR/AFR -> three runs
    assert chr1[["start", "end"]].values.tolist() == [[10000, 49999], [50000, 89999], [90000, 169999]]
    assert chr1[["Sample_1_EUR", "Sample_1_AFR"]].values.tolist() == [[2, 0], [1, 1], [0, 2]]
    chr2 = bed[bed["chromosome"] == "chr2"]
    assert chr2[["start", "end"]].values.tolist() == [[20000, 179999]]
    assert bed.equals(ds.la.to_bed(0))
    with pytest.raises(KeyError):
        to_bed(ds, "nobody")
    with pytest.raises(IndexError):
        to_bed(ds, 7)


def test_to_bed_min_segment():
    counts = np.zeros((10, 1, 2), dtype=np.int8)
    counts[:, 0, :] = [2, 0]
    counts[4, 0, :] = [1, 1]                      # one-variant blip
    loci = pd.DataFrame({"chromosome": ["chr1"] * 10, "physical_position": np.arange(10) * 100})
    ds = from_legacy(loci, None, counts)
    assert len(to_bed(ds, 0)) == 3
    bed = to_bed(ds, 0, min_segment=2)
    assert len(bed) == 1 and bed.loc[0, ["start", "end"]].tolist() == [0, 900]


# ------------------------------------------------------------------ positions
def test_at_positions_aggregate_and_sample_level(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    loci = pd.DataFrame({
        "variant_id": ["inside", "boundary", "outside", "other_chrom"],
        "chrom": ["1", "chr1", "chr1", "chr9"],
        "pos": [10001, 49999, 90000 + 80000, 5],
    })
    out = at_positions(ds, loci)
    assert out["variant_id"].tolist() == loci["variant_id"].tolist()
    assert out["matched"].tolist() == [True, True, False, False]
    # segment 0: S1 EUR/EUR, S2 EUR/AFR, S3 AFR/AFR -> EUR 3, AFR 3
    assert out.loc[0, "EUR_haplotypes"] == 3 and out.loc[0, "AFR_haplotypes"] == 3
    assert out.loc[0, "n_haplotypes"] == 6 and out.loc[0, "n_samples"] == 3
    assert out.loc[1, "EUR_fraction"] == pytest.approx(0.5)
    assert np.isnan(out.loc[2, "EUR_fraction"]) and out.loc[2, "n_haplotypes"] == 0

    per = at_positions(ds, loci.iloc[:1], samples=["Sample_3"], aggregate=False)
    assert per.shape[0] == 1 and per.loc[0, "sample_id"] == "Sample_3"
    assert per.loc[0, "EUR_copies"] == 0 and per.loc[0, "AFR_copies"] == 2

    allp = at_positions(ds, loci, aggregate=False)
    assert allp.shape[0] == 12 and np.isnan(allp.loc[allp["variant_id"] == "outside", "EUR_copies"]).all()

    with pytest.raises(ValueError, match="Samples not found"):
        at_positions(ds, loci, samples=["nope"])
    with pytest.raises(ValueError, match="must contain"):
        at_positions(ds, loci.drop(columns="pos"))
    assert at_positions(ds, loci.iloc[:0]).empty


def test_at_positions_nearest(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    loci = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [200000, 60000]})
    out = at_positions(ds, loci, method="nearest")
    assert out["matched"].tolist() == [True, True]
    assert out.loc[0, "AFR_haplotypes"] == 3   # nearest is the last segment (130000)


# ------------------------------------------------------------------ parquet
def test_to_parquet_streaming(msp_dir, tmp_path):
    pytest.importorskip("pyarrow")
    ds = open_rfmix(str(msp_dir), verbose=False)
    files = to_parquet(ds, tmp_path, prefix="la", rows_per_file=3)
    assert [f.name for f in files] == ["la.chr1-0.parquet", "la.chr1-1.parquet", "la.chr2-0.parquet",
                                       "la.chr2-1.parquet"]
    chr1 = pd.concat([pd.read_parquet(f) for f in files[:2]], ignore_index=True)
    names = ["Sample_1_EUR", "Sample_1_AFR", "Sample_2_EUR", "Sample_2_AFR", "Sample_3_EUR", "Sample_3_AFR"]
    assert list(chr1.columns) == ["chrom", "pos", "hap", *names]
    assert chr1["hap"].iloc[0] == "chr1_10000"
    np.testing.assert_array_equal(chr1[names].to_numpy(), ds.la.counts.values[:4].reshape(4, -1))


# ------------------------------------------------------------------ interpolation
def test_build_variant_grid(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    variants = pd.DataFrame({"chrom": ["1", "1", "chr2", "chr7"], "pos": [12000, 50000, 25000, 1]})
    grid = build_variant_grid(ds, variants)
    assert grid["chrom"].tolist() == ["chr1"] * 5 + ["chr2"] * 5
    assert grid.loc[grid["pos"] == 12000, "i"].isna().all()
    assert grid.loc[grid["pos"] == 50000, "i"].iloc[0] == 1
    assert build_variant_grid(ds, variants, include_source=False).shape[0] == 3


def test_interpolate_stepwise_and_linear(msp_dir, tmp_path):
    ds = open_rfmix(str(msp_dir), verbose=False)
    variants = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "pos": [12000, 95000, 25000]})
    out = interpolate(ds, variants, tmp_path / "z", method="stepwise", chunk_size=4)
    assert tuple(out.dims) == ("variant", "sample", "ancestry")
    assert out.sizes["variant"] == 11
    vals = out.values
    # 12000 lies in segment 0 (EUR/EUR, EUR/AFR, AFR/AFR); 95000 in segment 2
    pos = out.variant_position.values
    np.testing.assert_array_equal(vals[pos == 12000][0], [[2, 0], [1, 1], [0, 2]])
    np.testing.assert_array_equal(vals[pos == 95000][0], ds.la.counts.values[2])
    assert not np.isnan(vals).any()
    assert ds.la.interpolate(variants, tmp_path / "z2", method="linear").sizes["variant"] == 11


# ------------------------------------------------------------------ tagore
def test_to_tagore(msp_dir):
    pytest.importorskip("matplotlib")
    ds = open_rfmix(str(msp_dir), verbose=False)
    out = ds.la.to_tagore("Sample_1")
    for col in ("#chr", "start", "stop", "feature", "size", "color", "chrCopy"):
        assert col in out.columns
    bed = ds.la.to_bed("Sample_1")
    assert out.shape[0] == int(bed[["Sample_1_EUR", "Sample_1_AFR"]].to_numpy().sum())


# ------------------------------------------------------------------ zarr-backed inputs
def test_ops_on_cached_dataset(msp_dir, tmp_path):
    """Operations must accept the string dtypes a Zarr-backed Dataset returns."""
    pytest.importorskip("pyarrow")
    from rfmix_reader import open_local_ancestry

    mem = open_rfmix(str(msp_dir), verbose=False)
    open_rfmix(str(msp_dir), cache_dir=tmp_path / "c", verbose=False)
    ds = open_local_ancestry(tmp_path / "c")

    pd.testing.assert_frame_equal(to_bed(ds, "Sample_1"), to_bed(mem, "Sample_1"))
    loci = pd.DataFrame({"chrom": ["chr1", "chr2"], "pos": [10001, 25000]})
    pd.testing.assert_frame_equal(at_positions(ds, loci), at_positions(mem, loci))
    files = to_parquet(ds, tmp_path / "pq", prefix="la")
    assert [f.name for f in files] == ["la.chr1-0.parquet", "la.chr2-0.parquet"]
    out = interpolate(ds, loci, tmp_path / "z", method="stepwise")
    assert out.sizes["variant"] == 10 and out.chromosome.values.tolist()[:5] == ["chr1"] * 5
    assert ds.la.to_legacy()[0]["chromosome"].astype(str).tolist() == mem.la.to_legacy()[0]["chromosome"].astype(str).tolist()
