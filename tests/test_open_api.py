"""``open_*`` / ``convert`` / ``open_local_ancestry`` on the fixtures."""
import os
import time

import numpy as np
import pandas as pd
import pytest

import rfmix_reader
from rfmix_reader import convert, open_flare, open_local_ancestry, open_rfmix, open_simu
from tests.test_formats import FLARE_EXPECTED, SIMU_EXPECTED

MSP_CHR1_COUNTS = np.array([
    [[2, 0], [1, 1], [0, 2]],
    [[1, 1], [1, 1], [0, 2]],
    [[0, 2], [2, 0], [1, 1]],
    [[0, 2], [1, 1], [2, 0]],
], dtype=np.int8)


def _assert_same(ds, ref):
    """Two Datasets carry the same variants, counts and global ancestry."""
    np.testing.assert_array_equal(ds.la.counts.values, ref.la.counts.values)
    assert ds.variant_position.values.tolist() == ref.variant_position.values.tolist()
    assert ds.la.chromosomes == ref.la.chromosomes
    assert ds.la.samples == ref.la.samples and ds.la.ancestries == ref.la.ancestries
    pd.testing.assert_frame_equal(ds.la.global_ancestry, ref.la.global_ancestry)


def test_top_level_exports():
    for name in ("open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert",
                 "build_dataset", "from_legacy", "PhasingConfig", "phase_dataset"):
        assert callable(getattr(rfmix_reader, name))


def test_open_rfmix_msp_in_memory(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    assert ds.attrs["source_format"] == "msp"
    assert ds.la.chromosomes == ["chr1", "chr2"] and ds.sizes["contig"] == 2
    assert ds.segment_end.values[0] == 49999
    assert ds.la.ancestries == ["EUR", "AFR"]     # header order EUR=0 AFR=1
    np.testing.assert_array_equal(ds.la.counts.values[:4], MSP_CHR1_COUNTS)
    g = ds.la.global_ancestry
    assert list(g.columns) == ["sample_id", "EUR", "AFR", "chrom"]
    assert g["chrom"].tolist() == ["chr1"] * 3 + ["chr2"] * 3
    loci, g2, arr = ds.la.to_legacy()
    assert loci["i"].tolist() == list(range(8)) and arr.dtype == np.int8


def test_open_rfmix_msp_cache_and_reuse(msp_dir, tmp_path):
    ref = open_rfmix(str(msp_dir), verbose=False)
    cache = tmp_path / "cache"
    ds = open_rfmix(str(msp_dir), cache_dir=cache, verbose=False)
    assert sorted(p.name for p in cache.iterdir()) == ["chr1.zarr", "chr2.zarr"]
    _assert_same(ds, ref)

    stamp = os.path.getmtime(cache / "chr1.zarr" / "zarr.json")
    time.sleep(0.05)
    ds2 = open_rfmix(str(msp_dir), cache_dir=cache, verbose=False)    # reused, not rebuilt
    assert os.path.getmtime(cache / "chr1.zarr" / "zarr.json") == stamp
    _assert_same(ds2, ref)

    open_rfmix(str(msp_dir), cache_dir=cache, overwrite=True, verbose=False)
    assert os.path.getmtime(cache / "chr1.zarr" / "zarr.json") != stamp

    lazy = open_local_ancestry(cache)
    assert hasattr(lazy.haplotype_ancestry.data, "dask")
    _assert_same(lazy, ref)
    _assert_same(open_local_ancestry(cache, chrom="2"), open_rfmix(str(msp_dir), chrom="2", verbose=False))
    with pytest.raises(FileNotFoundError):
        open_local_ancestry(cache, chrom="22")
    with pytest.raises(FileNotFoundError):
        open_local_ancestry(tmp_path / "empty")


def test_open_rfmix_chrom_and_prefix(msp_dir):
    ds = open_rfmix(str(msp_dir), chrom="chr2", verbose=False)
    assert ds.la.chromosomes == ["chr2"] and ds.sizes["variant"] == 4
    ds = open_rfmix(str(msp_dir / "chr1"), verbose=False)
    assert ds.la.chromosomes == ["chr1"]
    with pytest.raises(FileNotFoundError):
        open_rfmix(str(msp_dir), chrom="9", verbose=False)
    with pytest.raises(ValueError):
        open_rfmix(str(msp_dir), source="bin", verbose=False)


def test_open_rfmix_fb_counts_and_posteriors(fb_dir, tmp_path):
    text = pd.read_csv(fb_dir / "chr1.fb.tsv", sep="\t", skiprows=1)
    post = text.iloc[:, 4:].to_numpy(np.float32).reshape(5, 4, 2, 2)
    expected = np.eye(2, dtype=np.int8)[post.argmax(-1)].sum(2)
    expected[~(post > 0).any(-1).all(-1)] = -1

    ds = open_rfmix(str(fb_dir), source="fb", verbose=False)
    assert ds.la.posterior is None and ds.la.ancestries == ["EUR", "AFR"]
    np.testing.assert_array_equal(ds.la.counts.values, expected)
    assert ds.variant_position.values.tolist() == text["physical_position"].tolist()
    assert ds.la.global_ancestry["EUR"].dtype == np.float32

    with pytest.warns(UserWarning, match="cache_dir"):
        ds = open_rfmix(str(fb_dir), source="fb", keep_posteriors=True, verbose=False)
    np.testing.assert_allclose(ds.la.posterior.values, post, atol=1e-6)

    ds = open_rfmix(str(fb_dir), source="fb", keep_posteriors=True, cache_dir=tmp_path / "c",
                    chunk_rows=2, verbose=False)
    assert ds.la.posterior.data.chunks[0] == (2, 2, 1)
    np.testing.assert_allclose(ds.la.posterior.values, post, atol=1e-6)
    np.testing.assert_array_equal(ds.la.counts.values, expected)


def test_open_flare(flare_dir, tmp_path):
    ds = open_flare(str(flare_dir), verbose=False)
    np.testing.assert_array_equal(ds.la.counts.values, FLARE_EXPECTED)
    assert ds.la.ancestries == ["EUR", "AFR"] and ds.la.chromosomes == ["chr21"]
    assert ds.variant_position.values.tolist() == [5030578, 5030588, 5031000, 5032000]
    np.testing.assert_allclose(ds.la.global_ancestry[["EUR", "AFR"]].to_numpy(),
                               [[0.625, 0.375], [0.375, 0.625]])
    cached = open_flare(str(flare_dir), cache_dir=tmp_path, chunk_rows=3, verbose=False)
    assert (tmp_path / "chr21.zarr").is_dir()
    assert cached.haplotype_ancestry.data.chunks[0] == (3, 1)
    _assert_same(cached, ds)


def test_open_simu_and_convert(simu_dir, tmp_path):
    ds = open_simu(str(simu_dir), verbose=False)
    np.testing.assert_array_equal(ds.la.counts.values, SIMU_EXPECTED)
    assert ds.la.ancestries == ["CEU", "NAT", "YRI"]
    assert ds.variant_position.values.tolist() == [100, 5000, 1500000]
    g = ds.la.global_ancestry
    np.testing.assert_allclose(g[["CEU", "NAT", "YRI"]].sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(g.loc[0, ["CEU", "NAT", "YRI"]].to_numpy(dtype=float), [0.5, 0, 0.5])

    paths = convert(str(simu_dir), "haptools", tmp_path, verbose=False)
    assert [p.name for p in paths] == ["chr21.zarr"]
    _assert_same(open_local_ancestry(tmp_path), ds)
    # a second convert without overwrite skips the existing store
    stamp = os.path.getmtime(paths[0] / "zarr.json")
    assert convert(str(simu_dir), "haptools", tmp_path, verbose=False) == paths
    assert os.path.getmtime(paths[0] / "zarr.json") == stamp


def test_convert_msp_two_chromosomes_then_open(msp_dir, tmp_path):
    paths = convert(str(msp_dir), "msp", tmp_path / "z", verbose=False)
    assert [p.name for p in paths] == ["chr1.zarr", "chr2.zarr"]
    ds = open_local_ancestry(tmp_path / "z")
    assert ds.sizes["variant"] == 8
    assert ds.la.global_ancestry["chrom"].tolist() == ["chr1"] * 3 + ["chr2"] * 3
    # counts are lazy: reading one chunk does not touch the rest
    sub = ds.la.counts[:2].compute()
    np.testing.assert_array_equal(sub[0], [[2, 0], [1, 1], [0, 2]])
