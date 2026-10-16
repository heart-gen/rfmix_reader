"""
Regression checks on the real chr21 RFMix output (git-LFS).  Run with

    pytest --run-slow tests/test_chr21_slow.py
"""
import time

import numpy as np
import pandas as pd
import pytest

from rfmix_reader import open_flare, open_local_ancestry, open_rfmix
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


@pytest.mark.slow
def test_chr21_logical_export(chr21_ds, chr21_lfs, tmp_path):
    """
    Export one megabase against the haptools genotypes and check the
    assumption the bundle rests on: Dataset haplotype 1 is the first allele of
    the VCF genotype.  The haptools ``POP`` field carries the simulated truth
    per haplotype in VCF order, so label agreement must be far higher
    haplotype-to-haplotype than crosswise.
    """
    from cyvcf2 import VCF
    pq = pytest.importorskip("pyarrow.parquet")
    from rfmix_reader.ops.logical import closed_form_frequencies, read_manifest, to_logical

    ds, _ = chr21_ds
    region = (15_000_000, 16_000_000)
    t0 = time.time()
    bundle = to_logical(ds, chr21_lfs / "chr21.vcf.gz", tmp_path, region=region, sites_per_file=5_000)
    print(f"chr21 LoGicAl export of 1 Mb: {time.time() - t0:.1f}s")
    m = read_manifest(bundle)
    assert m["samples"]["n_matched"] == 500 and m["K"] == 2 and m["has_posterior"]
    n_biallelic = sum(1 for r in VCF(str(chr21_lfs / "chr21.vcf.gz"))(f"chr21:{region[0]}-{region[1]}")
                      if len(r.ALT) == 1)
    assert m["sites"]["n_exported"] + m["sites"]["skipped"].get("all_missing", 0) == n_biallelic
    assert m["sites"]["distance"]["p50"] < 2_000

    sites = pd.concat([pq.read_table(bundle / f"sites-{f['k']}.parquet").to_pandas()
                       for f in m["layout"]["files"]], ignore_index=True)
    n = 500
    labels = ds.la.haplotypes.isel(variant=sites["window"].to_numpy()).values   # (sites, 500, 2) 0=AFR 1=EUR
    truth = np.full(labels.shape, -1, dtype=np.int8)
    pos_to_row = {p: i for i, p in enumerate(sites["pos"].tolist())}
    code = {"YRI": 0, "CEU": 1}
    for rec in VCF(str(chr21_lfs / "chr21.vcf.gz"))(f"chr21:{region[0]}-{region[1]}"):
        i = pos_to_row.get(rec.POS)
        if i is None:
            continue
        pops = rec.format("POP")
        for s in range(n):
            a, b = str(pops[s]).split(",")
            truth[i, s] = (code[a], code[b])
    valid = (labels >= 0).all(axis=2) & (truth >= 0).all(axis=2)
    het = valid & (truth[..., 0] != truth[..., 1])        # a swap is only visible where the two differ
    same = ((labels[..., 0] == truth[..., 0]) & (labels[..., 1] == truth[..., 1]))[het].mean()
    cross = ((labels[..., 0] == truth[..., 1]) & (labels[..., 1] == truth[..., 0]))[het].mean()
    print(f"haplotype-order agreement on {int(het.sum())} ancestry-heterozygous cells: "
          f"same {same:.3f}, crosswise {cross:.3f}")
    assert same > 0.9 and cross < 0.1

    cf = closed_form_frequencies(bundle, min_haplotypes=50)
    geno = pd.concat([pq.read_table(bundle / f"genotypes-{f['k']}.parquet", columns=["h1", "h2"]).to_pandas()
                      for f in m["layout"]["files"]], ignore_index=True)
    h = np.stack([geno["h1"].to_numpy().reshape(-1, n), geno["h2"].to_numpy().reshape(-1, n)], axis=-1)
    for a, anc in enumerate(["AFR", "EUR"]):
        mask = (truth == a) & (h >= 0)
        den = mask.sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            true_freq = (mask & (h == 1)).sum(axis=(1, 2)) / den
        ok = (den >= 50) & cf[f"{anc}_freq"].notna().to_numpy()
        r = np.corrcoef(true_freq[ok], cf.loc[ok, f"{anc}_freq"])[0, 1]
        print(f"{anc}: r = {r:.4f} over {ok.sum()} sites")
        assert r > 0.95


@pytest.mark.slow
def test_chr21_flare_matches_truth(chr21_lfs, tmp_path):
    """
    The bundled FLARE run must agree with the haptools truth: FLARE's global AFR
    tracks the simulated YRI fraction, both as written by FLARE and as rebuilt
    from the per-haplotype AN1/AN2 calls the reader decodes.  Guards against a
    FLARE fixture produced with a bad model (an inherited chr1 model with
    ``em=false`` once put every haplotype in EUR).
    """
    from rfmix_reader.formats.global_ancestry import fractions_from_counts, read_flare_global

    truth = pd.read_csv(chr21_lfs / "chr21.global.truth.tsv.gz", sep="\t", index_col="Sample")["YRI"]
    flare = read_flare_global(str(chr21_lfs / "chr21.global.anc.gz")).set_index("sample_id")["AFR"]
    common = flare.index.intersection(truth.index)
    assert len(common) == 500
    r = np.corrcoef(flare[common], truth[common])[0, 1]
    gap = flare[common].mean() - truth[common].mean()
    print(f"\nFLARE global.anc vs truth: r = {r:.3f}, mean AFR {flare[common].mean():.3f} "
          f"vs YRI {truth[common].mean():.3f}")
    assert r > 0.75 and abs(gap) < 0.05

    t0 = time.time()
    ds = open_flare(str(chr21_lfs), cache_dir=tmp_path, verbose=False)
    counts = ds.la.counts
    totals = counts.where(counts >= 0, 0).sum("variant").values           # (samples, K)
    afr = pd.Series(fractions_from_counts(totals)[:, ds.la.ancestries.index("AFR")],
                    index=ds.la.samples)
    # FLARE's global.anc comes from ancestry probabilities, the reader's from hard calls
    np.testing.assert_allclose(afr[common], flare[common], atol=0.05)
    r_reader = np.corrcoef(afr[common], truth[common])[0, 1]
    print(f"FLARE via reader ({time.time() - t0:.1f}s): r = {r_reader:.3f}")
    assert r_reader > 0.75
