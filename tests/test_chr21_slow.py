"""
Regression checks on the real chr21 RFMix output (git-LFS).  Run with

    pytest --run-slow tests/test_chr21_slow.py
"""
import time

import numpy as np
import pandas as pd
import pytest

from rfmix_reader import open_local_ancestry, open_rfmix
from rfmix_reader.processing.phase import PhasingConfig


@pytest.fixture(scope="module")
def chr21_ds(tmp_path_factory, chr21_lfs):
    """chr21 .fb.tsv converted once (with posteriors) into a module-scoped cache."""
    cache = tmp_path_factory.mktemp("cache")
    t0 = time.time()
    ds = open_rfmix(str(chr21_lfs), source="fb", keep_posteriors=True, cache_dir=cache, verbose=False)
    print(f"\nchr21 fb -> zarr (with posteriors): {time.time() - t0:.1f}s")
    return ds, cache


@pytest.mark.slow
def test_chr21_fb_counts_are_consistent(chr21_ds, chr21_lfs):
    """Counts equal the posterior argmax; sentinel rows are the all-zero haplotypes."""
    ds, cache = chr21_ds
    assert ds.la.ancestries == ["AFR", "EUR"] and ds.la.n_samples == 500
    q = pd.read_csv(chr21_lfs / "chr21.rfmix.Q", sep="\t", skiprows=1)
    np.testing.assert_allclose(ds.la.global_ancestry[["AFR", "EUR"]].to_numpy(),
                               q[["AFR", "EUR"]].to_numpy(), atol=1e-5)

    counts = ds.la.counts.values
    valid = counts[:, :, 0] >= 0
    assert (counts[valid].sum(axis=1) == 2).all()
    assert 0 < (~valid).mean() < 0.05       # ~0.7% all-zero haplotypes in this file

    post = ds.la.posterior[1000:1200].values
    assert set(np.unique(post).tolist()) <= {0.0, 1.0}
    assert np.isin(post.sum(axis=-1), [0.0, 1.0]).all()
    rows = np.flatnonzero(~valid.all(axis=1))[:3]
    for r in rows:
        no_mass = ~(ds.la.posterior[r].values > 0).any(axis=-1)
        np.testing.assert_array_equal(~valid[r], no_mass.any(axis=1))

    t0 = time.time()
    lazy = open_local_ancestry(cache)
    assert time.time() - t0 < 2.0
    assert lazy.sizes["variant"] == ds.sizes["variant"]
    store_bytes = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    print(f"chr21 store with posteriors: {store_bytes / 1e6:.1f} MB")


@pytest.mark.slow
def test_chr21_gnomix_phasing_on_posteriors(chr21_ds):
    """gnomix-style phasing of the whole chromosome from the RFMix posteriors."""
    ds, _ = chr21_ds
    t0 = time.time()
    phased = ds.la.phase(config=PhasingConfig(window_size=50, min_block_len=20))
    swapped = phased["phase_swapped"].values
    phase_s = time.time() - t0
    assert swapped.shape == (ds.la.n_variants, 500)
    np.testing.assert_array_equal(phased.la.counts.values, ds.la.counts.values)
    frac = swapped.mean()
    print(f"chr21 gnomix phasing: {phase_s:.1f}s, {frac:.2%} of sample-loci exchanged")
    assert 0 <= frac < 0.5


@pytest.mark.slow
def test_chr21_ops_scale(chr21_ds, tmp_path):
    """BED, position queries and Parquet export on a whole chromosome."""
    ds, _ = chr21_ds
    t0 = time.time()
    bed = ds.la.to_bed("Sample_2", min_segment=3)      # Sample_2 is admixed (AFR 0.80 / EUR 0.20)
    assert len(bed) > 1 and (bed["end"] >= bed["start"]).all()
    assert bed["Sample_2_AFR"].nunique() > 1
    assert len(ds.la.to_bed("Sample_1")) == 1           # Sample_1 is AFR/AFR everywhere
    q = pd.DataFrame({"chrom": ["chr21"] * 3, "pos": [5030578, 20_000_000, 46_000_000]})
    hit = ds.la.at_positions(q, method="nearest")
    assert hit["matched"].all() and hit["n_haplotypes"].max() <= 1000
    files = ds.la.to_parquet(tmp_path, prefix="la", rows_per_file=50_000)
    assert len(files) == 4
    print(f"chr21 bed/positions/parquet: {time.time() - t0:.1f}s")
