"""The deprecated public names route through the Dataset core and keep their results."""
import numpy as np
import pandas as pd
import pytest

import rfmix_reader
from rfmix_reader.readers.read_flare import read_flare as module_read_flare
from rfmix_reader.readers.read_msp import extract_locus_ancestry as module_extract
from rfmix_reader.readers.read_msp import read_rfmix as module_read_rfmix
from rfmix_reader.readers.read_simu import read_simu as module_read_simu


def _same_triple(a, b):
    la, ga, aa = a
    lb, gb, ab = b
    np.testing.assert_array_equal(aa.compute(), ab.compute())
    assert la["physical_position"].tolist() == lb["physical_position"].tolist()
    assert la["chromosome"].astype(str).tolist() == lb["chromosome"].astype(str).tolist()
    if gb is not None:
        assert list(ga.columns) == list(gb.columns)
        pops = [c for c in gb.columns if c not in ("sample_id", "chrom")]
        np.testing.assert_allclose(ga[pops].to_numpy(dtype=float), gb[pops].to_numpy(dtype=float), atol=1e-5)


def test_read_rfmix_wrapper_warns_and_matches(msp_dir):
    with pytest.warns(DeprecationWarning, match="open_rfmix"):
        out = rfmix_reader.read_rfmix(str(msp_dir), verbose=False)
    _same_triple(out, module_read_rfmix(str(msp_dir), verbose=False))
    with pytest.warns(DeprecationWarning):
        _, g, _ = rfmix_reader.read_rfmix(str(msp_dir), verbose=False, read_q=False)
    assert g is None
    custom = pd.DataFrame({"sample_id": ["a"], "AFR": [0.5], "EUR": [0.5]})
    with pytest.warns(DeprecationWarning):
        _, g, _ = rfmix_reader.read_rfmix(str(msp_dir), g_anc=custom, verbose=False)
    assert list(g.columns) == ["sample_id", "EUR", "AFR"]


def test_read_rfmix_fb_wrapper(fb_dir, tmp_path):
    with pytest.warns(DeprecationWarning, match='source="fb"'):
        loci, g_anc, admix, X_raw = rfmix_reader.read_rfmix_fb(
            str(fb_dir), binary_dir=str(tmp_path / "cache"), generate_binary=True,
            verbose=False, return_original=True)
    assert (tmp_path / "cache" / "chr1.zarr").is_dir()
    assert admix.dtype == np.int8 and admix.shape == (5, 4, 2)
    assert X_raw.shape == (5, 16) and X_raw.dtype == np.float32
    text = pd.read_csv(fb_dir / "chr1.fb.tsv", sep="\t", skiprows=1).iloc[:, 4:].to_numpy(np.float32)
    np.testing.assert_allclose(X_raw.compute(), text, atol=1e-6)
    # a later call without generate_binary reuses the cache
    with pytest.warns(DeprecationWarning):
        out = rfmix_reader.read_rfmix_fb(str(fb_dir), binary_dir=str(tmp_path / "cache"), verbose=False)
    assert len(out) == 3 and np.array_equal(out[2].compute(), admix.compute())


def test_read_flare_and_simu_wrappers(flare_dir, simu_dir):
    with pytest.warns(DeprecationWarning, match="open_flare"):
        out = rfmix_reader.read_flare(str(flare_dir), verbose=False)
    _same_triple(out, module_read_flare(str(flare_dir), verbose=False))
    with pytest.warns(DeprecationWarning, match="open_simu"):
        out = rfmix_reader.read_simu(str(simu_dir), verbose=False)
    _same_triple(out, module_read_simu(str(simu_dir), verbose=False))


def test_extract_locus_ancestry_wrapper_matches_module(msp_dir):
    loci = pd.DataFrame({"variant_id": ["a", "b", "c"], "chrom": ["1", "chr1", "chr1"],
                         "pos": [10001, 49999, 500000]})
    with pytest.warns(DeprecationWarning, match="at_positions"):
        new = rfmix_reader.extract_locus_ancestry(str(msp_dir), loci)
    old = module_extract(str(msp_dir), loci)
    assert new["matched"].tolist() == old["matched"].tolist()
    for col in ("n_samples", "n_haplotypes", "EUR_haplotypes", "AFR_haplotypes", "EUR_fraction"):
        np.testing.assert_allclose(new[col].to_numpy(dtype=float), old[col].to_numpy(dtype=float))
    with pytest.warns(DeprecationWarning):
        new_s = rfmix_reader.extract_locus_ancestry(str(msp_dir), loci.iloc[:2], samples=["Sample_2"],
                                                    aggregate=False)
    old_s = module_extract(str(msp_dir), loci.iloc[:2], samples=["Sample_2"], aggregate=False)
    assert new_s["EUR_copies"].tolist() == old_s["EUR_copies"].tolist()


def test_write_data_and_bed_wrappers(msp_dir, tmp_path):
    pytest.importorskip("pyarrow")
    loci, g_anc, admix = module_read_rfmix(str(msp_dir), verbose=False)
    with pytest.warns(DeprecationWarning, match="to_parquet"):
        rfmix_reader.write_data(loci, g_anc, admix, outdir=str(tmp_path), prefix="la")
    df = pd.read_parquet(tmp_path / "la.chr1-0.parquet")
    assert df.shape == (4, 9) and df["hap"].iloc[0] == "chr1_10000"

    with pytest.warns(DeprecationWarning, match="to_bed"):
        bed = rfmix_reader.admix_to_bed_individual(loci, g_anc, admix, 0, min_segment=1)
    assert list(bed.columns) == ["chromosome", "start", "end", "Sample_1_EUR", "Sample_1_AFR"]
    assert len(bed) == 4

    pytest.importorskip("matplotlib")
    with pytest.warns(DeprecationWarning, match="to_tagore"):
        tag = rfmix_reader.generate_tagore_bed(loci, g_anc, admix, 0, min_segment=1)
    assert "#chr" in tag.columns


def test_create_binaries_wrapper_warns(fb_dir, tmp_path):
    with pytest.warns(DeprecationWarning, match="convert"):
        rfmix_reader.create_binaries(str(fb_dir), str(tmp_path), verbose=False)
    assert (tmp_path / "chr1.bin").exists()
