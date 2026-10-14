"""
LoGicAl export: genotype reader, site-to-window mapping, bundle layout,
missing-value encoding, phase swaps and the closed-form reference.
"""
import json

import numpy as np
import pandas as pd
import pytest

from rfmix_reader import open_rfmix
from rfmix_reader.io.genotypes import gl_to_gl4, gt_to_calls, iter_genotype_blocks, pl_to_gl4
from rfmix_reader.ops.logical import MANIFEST_KEY, closed_form_frequencies, read_manifest, to_logical

pq = pytest.importorskip("pyarrow.parquet")

# chr1.vcf.gz next to the .fb fixture: sites that survive the export, in order,
# with the CRF point they map to (nearest) and the distance.
FB_SITES = [  # pos, id, window, distance
    (5030578, "rs1", 0, 0), (5031000, "rs2", 1, 40), (5031340, "rs3", 3, 115),
    (5031600, "rs4", 3, 145), (5031900, "rs6", 4, 100), (5032000, "rs7", 4, 0),
    (5040000, "rs9", 4, 8000),
]


@pytest.fixture(scope="module")
def fb_ds(tmp_path_factory, fb_dir):
    cache = tmp_path_factory.mktemp("cache")
    return open_rfmix(str(fb_dir), source="fb", keep_posteriors=True, cache_dir=cache,
                      chunk_rows=2, verbose=False)


@pytest.fixture(scope="module")
def fb_bundle(tmp_path_factory, fb_dir, fb_ds):
    out = tmp_path_factory.mktemp("bundle")
    return to_logical(fb_ds, fb_dir / "chr1.vcf.gz", out, sites_per_file=3)


def _read(bundle, name):
    return pq.read_table(bundle / f"{name}.parquet").to_pandas()


# ------------------------------------------------------------------ genotypes
def test_gt_to_calls():
    arr = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0], [-1, -1, 0], [0, 2, 1]])
    g, h1, h2 = gt_to_calls(arr)
    assert g.tolist() == [1, 1, 1, 2, -1, -1]
    assert h1.tolist() == [0, 1, -1, 1, -1, -1]
    assert h2.tolist() == [1, 0, -1, 1, -1, -1]
    assert g.dtype == np.int8 and h1.dtype == np.int8


def test_pl_and_gl_to_gl4():
    out = pl_to_gl4(np.array([[0, 10, 20], [-2147483648, 0, 0], [30, 0, 30]]))
    np.testing.assert_allclose(out[0], [1, 0.1, 0.1, 0.01])
    assert np.isnan(out[1]).all()
    np.testing.assert_allclose(out[2], [0.001, 1, 1, 0.001])
    assert out.dtype == np.float32
    out = gl_to_gl4(np.array([[0.0, -1.0, -2.0], [np.nan, 0.0, 0.0]]))
    np.testing.assert_allclose(out[0], [1, 0.1, 0.1, 0.01], rtol=1e-6)
    assert np.isnan(out[1]).all()
    with pytest.raises(ValueError):
        pl_to_gl4(np.zeros((3, 4)))


def test_iter_genotype_blocks(fb_dir):
    blocks = list(iter_genotype_blocks(fb_dir / "chr1.vcf.gz", chrom="1", block_size=4,
                                       samples=["Sample_4", "Sample_1"]))
    assert [len(b) for b in blocks] == [4, 3, 1]         # blocks of 4 records; one multi-allelic dropped
    assert blocks[0].samples == ["Sample_1", "Sample_4"]  # cyvcf2 keeps file order
    assert blocks[1].skipped == [(5031800, "A", "C,T", "multiallelic")]
    first = blocks[0]
    assert first.pos.tolist() == [5030578, 5031000, 5031340, 5031600]
    assert first.g[0].tolist() == [0, 1] and first.h1[0].tolist() == [0, 1] and first.h2[0].tolist() == [0, 0]
    assert first.gl.shape == (4, 2, 4) and np.isnan(first.gl[2]).all()   # rs3 has PL "."
    np.testing.assert_allclose(first.gl[0, 0], [1, 1e-3, 1e-3, 1e-6])
    with_region = list(iter_genotype_blocks(fb_dir / "chr1.vcf.gz", chrom="chr1",
                                            region=(5031000, 5031900), snps_only=True))
    assert with_region[0].pos.tolist() == [5031000, 5031340, 5031600]
    assert {r[3] for r in with_region[0].skipped} == {"multiallelic", "non_snp"}
    assert list(iter_genotype_blocks(fb_dir / "chr1.vcf.gz", chrom="chr2")) == []


# ------------------------------------------------------------------ bundle
def test_bundle_sites_and_mapping(fb_bundle):
    m = read_manifest(fb_bundle)
    assert m["format_version"] == 1 and m["K"] == 2 and m["ancestries"] == ["EUR", "AFR"]
    assert m["has_posterior"] and m["gl_source"] == "PL" and m["mapping"]["method"] == "nearest"
    sites = pd.concat([_read(fb_bundle, f"sites-{k}") for k in range(3)], ignore_index=True)
    assert [(p, i, w, d) for p, i, w, d in sites[["pos", "id", "window", "distance"]].itertuples(index=False)] == FB_SITES
    assert sites["site"].tolist() == list(range(7)) and sites["file"].tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert sites["n_called"].tolist() == [4, 4, 4, 4, 4, 3, 4]
    assert sites["n_phased"].tolist() == [4, 4, 4, 2, 4, 3, 4]
    assert sites["is_snp"].tolist() == [True] * 4 + [False] + [True] * 2
    skipped = _read(fb_bundle, "skipped_sites")
    assert sorted(zip(skipped["pos"], skipped["reason"])) == [(5031800, "multiallelic"), (5032100, "all_missing")]
    assert m["sites"] == {"n_vcf_records": 9, "n_exported": 7,
                          "skipped": {"multiallelic": 1, "all_missing": 1},
                          "distance": m["sites"]["distance"]}
    assert m["sites"]["distance"]["max"] == 8000
    assert [f["n_sites"] for f in m["layout"]["files"]] == [3, 2, 2]
    assert m["layout"]["files"][0] == {"k": 0, "n_sites": 3, "site_first": 0, "site_last": 2,
                                       "window_min": 0, "window_max": 3}
    windows = _read(fb_bundle, "windows")
    assert windows["pos"].tolist() == [5030578, 5030960, 5031200, 5031455, 5032000]
    assert (windows["end"] == windows["pos"]).all()
    assert (fb_bundle / "logical_run.R").is_file()


def test_max_distance_and_snps_only(fb_ds, fb_dir, tmp_path):
    b = to_logical(fb_ds, fb_dir / "chr1.vcf.gz", tmp_path, max_distance=1000, snps_only=True,
                   write_r_helper=False)
    m = read_manifest(b)
    assert m["sites"]["skipped"] == {"multiallelic": 1, "non_snp": 1, "all_missing": 1, "beyond_max_distance": 1}
    assert m["sites"]["n_exported"] == 5 and m["mapping"]["max_distance"] == 1000
    skipped = _read(b, "skipped_sites").set_index("pos")["reason"]
    assert skipped[5040000] == "beyond_max_distance" and skipped[5031900] == "non_snp"
    assert not (b / "logical_run.R").exists()


def test_labels_and_posteriors(fb_bundle, fb_ds):
    post = pd.concat([_read(fb_bundle, f"posterior-{j}") for j in range(3)], ignore_index=True)
    n, W = 4, 2
    assert read_manifest(fb_bundle)["layout"]["windows_per_file"] == W
    assert len(post) == 5 * n * 2
    # dense (window, sample, hap) order
    expected_rows = [(w, s, h) for w in range(5) for s in range(n) for h in (0, 1)]
    assert list(zip(post["window"], post["sample_idx"], post["hap"])) == expected_rows
    codes = fb_ds.la.haplotypes.values                       # (5, 4, 2)
    expected_label = np.where(codes >= 0, codes + 1, 0).reshape(-1)
    assert post["label"].tolist() == expected_label.tolist()
    assert post["label"].iloc[3 * n * 2 + 3 * 2 + 1] == 0        # Sample_4 hap2 at row 3: no mass
    p = fb_ds.la.posterior.values.reshape(-1, 2)
    np.testing.assert_allclose(post[["p_EUR", "p_AFR"]].to_numpy(), p, atol=1e-6)
    assert (post.loc[post["label"] == 0, ["p_EUR", "p_AFR"]].to_numpy() == 0).all()


def test_sample_intersection(fb_ds, fb_dir, tmp_path):
    m = read_manifest(to_logical(fb_ds, fb_dir / "chr1.vcf.gz", tmp_path / "a"))
    assert m["samples"] == {"n_dataset": 4, "n_vcf": 5, "n_matched": 4, "n_dropped_dataset": 0,
                            "n_dropped_vcf": 1, "dropped_dataset": [], "dropped_vcf": ["Extra_1"]}
    b = to_logical(fb_ds, fb_dir / "chr1.vcf.gz", tmp_path / "b", samples=["Sample_3", "Sample_1"])
    samples = _read(b, "samples")
    assert samples["sample_id"].tolist() == ["Sample_1", "Sample_3"]      # Dataset order
    assert samples["dataset_idx"].tolist() == [0, 2] and samples["vcf_idx"].tolist() == [0, 2]
    geno = _read(b, "genotypes-0")
    assert geno.loc[(geno["site"] == 0) & (geno["sample_idx"] == 1), "g"].item() == 1   # Sample_3 rs1 0|1
    with pytest.raises(ValueError, match="No sample"):
        to_logical(fb_ds, fb_dir / "chr1.vcf.gz", tmp_path / "c", samples=["nobody"])


def test_genotype_rows_and_metadata(fb_bundle):
    geno = _read(fb_bundle, "genotypes-0")
    n = 4
    assert len(geno) == 3 * n
    row = geno.iloc[0 * n + 3]                                   # site 0, Sample_4: 1|0
    assert (row["site"], row["sample_idx"], row["g"], row["h1"], row["h2"]) == (0, 3, 1, 1, 0)
    assert geno["g"].dtype == np.int8 and geno["h1"].dtype == np.int8
    np.testing.assert_allclose(geno.loc[0, ["gl_00", "gl_10", "gl_01", "gl_11"]].to_numpy(dtype=float),
                               [1, 1e-3, 1e-3, 1e-6])
    assert np.isnan(geno.loc[2 * n:, ["gl_00", "gl_11"]].to_numpy(dtype=float)).all()   # rs3: PL missing
    geno1 = _read(fb_bundle, "genotypes-1")
    rs4 = geno1.iloc[0:n]                                         # unphased site: 0/1 0/0 1/1 0/1
    assert rs4["g"].tolist() == [1, 0, 2, 1]
    assert rs4["h1"].tolist() == [-1, 0, 1, -1] and rs4["h2"].tolist() == [-1, 0, 1, -1]
    geno2 = _read(fb_bundle, "genotypes-2")
    assert geno2.iloc[1][["g", "h1", "h2"]].tolist() == [-1, -1, -1]                    # rs7 Sample_2 ./.
    for name in ("sites-0", "genotypes-0", "posterior-0", "samples", "windows", "skipped_sites"):
        meta = pq.read_table(fb_bundle / f"{name}.parquet").schema.metadata
        assert json.loads(meta[MANIFEST_KEY])["chromosome"] == "chr1"
        assert not any(k.startswith(b"r") for k in meta)       # R's arrow partial-matches metadata$r


def test_closed_form_frequencies(fb_bundle):
    cf = closed_form_frequencies(fb_bundle)
    assert cf["pos"].tolist() == [p for p, *_ in FB_SITES]
    rs1 = cf.iloc[0]
    # window 0 labels: S1 EUR/EUR, S2 AFR/AFR, S3 AFR/EUR, S4 AFR/EUR; rs1: 0|0 1|1 0|1 1|0
    assert (rs1["EUR_freq"], rs1["EUR_n"], rs1["AFR_freq"], rs1["AFR_n"]) == (0.25, 4, 0.75, 4)
    assert cf["n_haplotypes"].tolist() == [8, 8, 7, 4, 8, 6, 8]
    assert cf["n_used"].tolist() == [4, 4, 4, 2, 4, 3, 4]
    assert (cf["EUR_n"] + cf["AFR_n"] == cf["n_haplotypes"]).all()
    strict = closed_form_frequencies(fb_bundle, min_haplotypes=5)
    assert np.isnan(strict["EUR_freq"]).all() and strict["AFR_freq"].notna().sum() == 3   # AFR_n >= 5 at 3 sites


def test_phase_swaps_applied(fb_ds, fb_dir, tmp_path):
    mask = np.zeros((5, 4), dtype=bool)
    mask[0, 2] = True                                  # window 0, Sample_3
    mask[4, :] = True                                  # window 4, everyone
    ds = fb_ds.assign(phase_swapped=(("variant", "sample"), mask))
    b = to_logical(ds, fb_dir / "chr1.vcf.gz", tmp_path / "on")
    geno = _read(b, "genotypes-0")                     # default sites_per_file: one file
    rs1 = geno.iloc[0:4]
    assert rs1["h1"].tolist() == [0, 1, 1, 1] and rs1["h2"].tolist() == [0, 1, 0, 0]   # S3 0|1 -> 1|0
    assert read_manifest(b)["phase_swaps"] == {"applied": True, "n_swapped_cells": 1 + 4 * 3}
    assert _read(b, "sites-0")["window"].tolist() == [0, 1, 3, 3, 4, 4, 4]
    off = to_logical(ds, fb_dir / "chr1.vcf.gz", tmp_path / "off", apply_phase_swaps=False)
    assert _read(off, "genotypes-0").iloc[2]["h1"] == 0
    assert read_manifest(off)["phase_swaps"] == {"applied": False, "n_swapped_cells": 0}


def test_msp_dataset_exports_labels_only(msp_dir, tmp_path):
    ds = open_rfmix(str(msp_dir), source="msp", chrom="chr1", verbose=False)
    vcf = tmp_path / "chr1.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1,length=248956422>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1\tSample_2\tSample_3\n"
        "chr1\t10000\t.\tA\tG\t.\t.\t.\tGT\t0|1\t1|1\t0|0\n"
        "chr1\t55000\t.\tC\tT\t.\t.\t.\tGT\t1|0\t0|1\t1|1\n"
        "chr1\t169999\t.\tG\tA\t.\t.\t.\tGT\t0|0\t0|0\t0|1\n"
        "chr1\t170000\t.\tT\tC\t.\t.\t.\tGT\t0|0\t0|0\t0|1\n"
    )
    b = to_logical(ds, vcf, tmp_path / "bundle")
    m = read_manifest(b)
    assert m["has_posterior"] is False and m["gl_source"] is None and m["mapping"]["method"] == "stepwise"
    sites = _read(b, "sites-0")
    assert sites["pos"].tolist() == [10000, 55000, 169999]
    assert sites["window"].tolist() == [0, 1, 3] and sites["distance"].tolist() == [0, 0, 0]
    assert _read(b, "skipped_sites").to_dict("records") == [{"pos": 170000, "ref": "T", "alt": "C", "reason": "no_window"}]
    post = _read(b, "posterior-0")
    assert "p_EUR" not in post.columns and post["label"].min() >= 1
    geno = _read(b, "genotypes-0")
    assert "gl_00" not in geno.columns
    cf = closed_form_frequencies(b)
    # segment 0: S1 EUR/EUR, S2 EUR/AFR, S3 AFR/AFR; 10000: 0|1 1|1 0|0 -> EUR 2/3, AFR 1/3
    np.testing.assert_allclose(cf.iloc[0][["EUR_freq", "AFR_freq"]].to_numpy(dtype=float), [2 / 3, 1 / 3])


def test_fb_without_posteriors_warns(fb_dir, tmp_path):
    ds = open_rfmix(str(fb_dir), source="fb", verbose=False)
    with pytest.warns(UserWarning, match="hard labels only"):
        b = to_logical(ds, fb_dir / "chr1.vcf.gz", tmp_path)
    assert read_manifest(b)["has_posterior"] is False


def test_multi_chromosome_requires_chrom(msp_dir, tmp_path):
    ds = open_rfmix(str(msp_dir), source="msp", verbose=False)
    with pytest.raises(ValueError, match="single chromosome"):
        to_logical(ds, msp_dir / "nothing.vcf", tmp_path)
