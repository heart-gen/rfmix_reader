"""``open_*`` / ``convert`` / ``open_local_ancestry`` and equivalence with the legacy readers."""
import os
import time

import numpy as np
import pandas as pd
import pytest

import rfmix_reader
from rfmix_reader import convert, open_flare, open_local_ancestry, open_rfmix, open_simu
from rfmix_reader.readers.read_flare import read_flare
from rfmix_reader.readers.read_msp import read_rfmix
from rfmix_reader.readers.read_rfmix import read_rfmix_fb
from rfmix_reader.readers.read_simu import read_simu
from rfmix_reader.utils import create_binaries


def _assert_equivalent(ds, legacy):
    loci, g_anc, arr = ds.la.to_legacy()
    loci2, g2, arr2 = legacy
    np.testing.assert_array_equal(arr.compute(), arr2.compute())
    assert loci["physical_position"].tolist() == loci2["physical_position"].tolist()
    assert loci["chromosome"].astype(str).tolist() == loci2["chromosome"].astype(str).tolist()
    assert loci["i"].tolist() == loci2["i"].tolist()
    if g2 is not None:
        assert list(g_anc.columns) == list(g2.columns)
        assert g_anc["sample_id"].tolist() == g2["sample_id"].tolist()
        assert g_anc["chrom"].tolist() == g2["chrom"].tolist()
        pops = ds.la.ancestries
        np.testing.assert_allclose(g_anc[pops].to_numpy(dtype=float), g2[pops].to_numpy(dtype=float), atol=1e-5)


def test_top_level_exports():
    for name in ("open_rfmix", "open_flare", "open_simu", "open_local_ancestry", "convert"):
        assert callable(getattr(rfmix_reader, name))


def test_open_rfmix_msp_in_memory_matches_legacy(msp_dir):
    ds = open_rfmix(str(msp_dir), verbose=False)
    assert ds.attrs["source_format"] == "msp"
    assert ds.la.chromosomes == ["chr1", "chr2"] and ds.sizes["contig"] == 2
    assert ds.segment_end.values[0] == 49999
    _assert_equivalent(ds, read_rfmix(str(msp_dir), verbose=False))


def test_open_rfmix_msp_cache_and_reuse(msp_dir, tmp_path):
    cache = tmp_path / "cache"
    ds = open_rfmix(str(msp_dir), cache_dir=cache, verbose=False)
    assert sorted(p.name for p in cache.iterdir()) == ["chr1.zarr", "chr2.zarr"]
    _assert_equivalent(ds, read_rfmix(str(msp_dir), verbose=False))

    stamp = os.path.getmtime(cache / "chr1.zarr" / "zarr.json")
    time.sleep(0.05)
    ds2 = open_rfmix(str(msp_dir), cache_dir=cache, verbose=False)    # reused, not rebuilt
    assert os.path.getmtime(cache / "chr1.zarr" / "zarr.json") == stamp
    _assert_equivalent(ds2, read_rfmix(str(msp_dir), verbose=False))

    open_rfmix(str(msp_dir), cache_dir=cache, overwrite=True, verbose=False)
    assert os.path.getmtime(cache / "chr1.zarr" / "zarr.json") != stamp

    lazy = open_local_ancestry(cache)
    assert hasattr(lazy.haplotype_ancestry.data, "dask")
    _assert_equivalent(lazy, read_rfmix(str(msp_dir), verbose=False))
    _assert_equivalent(open_local_ancestry(cache, chrom="2"),
                       read_rfmix(str(msp_dir), verbose=False, chrom="2"))
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


def test_open_rfmix_fb_matches_legacy_and_keeps_posteriors(fb_dir, tmp_path):
    create_binaries(str(fb_dir), str(tmp_path / "bin"), verbose=False)
    legacy = read_rfmix_fb(str(fb_dir), binary_dir=str(tmp_path / "bin"), verbose=False)

    ds = open_rfmix(str(fb_dir), source="fb", verbose=False)
    assert ds.la.posterior is None
    _assert_equivalent(ds, legacy)

    with pytest.warns(UserWarning, match="cache_dir"):
        ds = open_rfmix(str(fb_dir), source="fb", keep_posteriors=True, verbose=False)
    text = pd.read_csv(fb_dir / "chr1.fb.tsv", sep="\t", skiprows=1).iloc[:, 4:]
    np.testing.assert_allclose(ds.la.posterior.values, text.to_numpy(np.float32).reshape(5, 4, 2, 2), atol=1e-6)

    ds = open_rfmix(str(fb_dir), source="fb", keep_posteriors=True, cache_dir=tmp_path / "c",
                    chunk_rows=2, verbose=False)
    assert ds.la.posterior.data.chunks[0] == (2, 2, 1)
    np.testing.assert_allclose(ds.la.posterior.values, text.to_numpy(np.float32).reshape(5, 4, 2, 2), atol=1e-6)
    _assert_equivalent(ds, legacy)


def test_open_flare_matches_legacy(flare_dir, tmp_path):
    _assert_equivalent(open_flare(str(flare_dir), verbose=False), read_flare(str(flare_dir), verbose=False))
    ds = open_flare(str(flare_dir), cache_dir=tmp_path, chunk_rows=3, verbose=False)
    assert (tmp_path / "chr21.zarr").is_dir()
    assert ds.haplotype_ancestry.data.chunks[0] == (3, 1)
    _assert_equivalent(ds, read_flare(str(flare_dir), verbose=False))


def test_open_simu_matches_legacy(simu_dir, tmp_path):
    _assert_equivalent(open_simu(str(simu_dir), verbose=False), read_simu(str(simu_dir), verbose=False))
    paths = convert(str(simu_dir), "haptools", tmp_path, verbose=False)
    assert [p.name for p in paths] == ["chr21.zarr"]
    _assert_equivalent(open_local_ancestry(tmp_path), read_simu(str(simu_dir), verbose=False))
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
