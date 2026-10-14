"""
The packaged R helper against the numpy reference.  Needs ``Rscript`` with
``arrow`` and ``jsonlite``; the EM methods also need SQUAREM and LoGicAl
(``remotes::install_github("student-jw/LoGicAl")``).
"""
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from rfmix_reader import open_rfmix
from rfmix_reader.core.schema import build_dataset
from rfmix_reader.ops.logical import closed_form_frequencies, to_logical

pytest.importorskip("pyarrow")


def _r_has(*packages: str) -> bool:
    if shutil.which("Rscript") is None:
        return False
    code = "; ".join(f"library({p})" for p in packages)
    try:
        proc = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _run_helper(bundle, method: str, out, min_n: int = 1) -> pd.DataFrame:
    proc = subprocess.run(
        ["Rscript", str(bundle / "logical_run.R"), "--bundle", str(bundle), "--method", method,
         "--out", str(out), "--min-n", str(min_n)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stderr, proc.stderr
    return pd.read_csv(out, sep="\t")


@pytest.fixture(scope="module")
def fb_bundle(tmp_path_factory, fb_dir):
    ds = open_rfmix(str(fb_dir), source="fb", keep_posteriors=True,
                    cache_dir=tmp_path_factory.mktemp("cache"), chunk_rows=2, verbose=False)
    return to_logical(ds, fb_dir / "chr1.vcf.gz", tmp_path_factory.mktemp("bundle"), sites_per_file=3)


@pytest.mark.skipif(not _r_has("arrow", "jsonlite"), reason="needs Rscript with arrow and jsonlite")
def test_r_phase_ag_equals_closed_form(fb_bundle, tmp_path):
    got = _run_helper(fb_bundle, "phase_ag", tmp_path / "out.tsv")
    ref = closed_form_frequencies(fb_bundle)
    assert got["pos"].tolist() == ref["pos"].tolist()
    assert got["n_used"].tolist() == ref["n_used"].tolist()
    assert got["n_haplotypes"].tolist() == ref["n_haplotypes"].tolist()
    assert (got["method"] == "phase_ag").all() and got["error"].isna().all()
    for anc in ("EUR", "AFR"):
        np.testing.assert_allclose(got[f"{anc}_freq"], ref[f"{anc}_freq"], atol=1e-12)
    strict = _run_helper(fb_bundle, "phase_ag", tmp_path / "strict.tsv", min_n=4)
    assert strict["error"].fillna("").tolist().count("below min-n") == 2       # rs4 (2 used) and rs7 (3 used)


@pytest.mark.skipif(not _r_has("arrow", "jsonlite", "SQUAREM", "LoGicAl"),
                    reason="needs the LoGicAl R package")
def test_r_em_methods_run(fb_bundle, tmp_path):
    for method in ("pg", "ag", "phase_pg", "pl", "al"):
        got = _run_helper(fb_bundle, method, tmp_path / f"{method}.tsv")
        ok = got["error"].isna()
        assert ok.sum() >= 5, got
        est = got.loc[ok, ["EUR_freq", "AFR_freq"]].to_numpy(dtype=float)
        assert np.isfinite(est).all() and (est >= 0).all() and (est <= 1).all()


@pytest.mark.skipif(not _r_has("arrow", "jsonlite", "SQUAREM", "LoGicAl"),
                    reason="needs the LoGicAl R package")
def test_r_phase_pg_on_hard_posteriors_equals_closed_form(fb_dir, tmp_path):
    """With 0/1 posteriors the phased EM's fixed point is the closed-form frequency."""
    ds = open_rfmix(str(fb_dir), source="fb", keep_posteriors=True,
                    cache_dir=tmp_path / "cache", chunk_rows=2, verbose=False)
    codes = ds.la.haplotypes.values
    onehot = np.zeros(codes.shape + (2,), dtype=np.float32)
    for a in range(2):
        onehot[..., a] = codes == a
    hard = build_dataset(ds.chromosome.values, ds.variant_position.values, ds.segment_end.values,
                         codes, ds.la.samples, ds.la.ancestries, posterior=onehot,
                         source_format="fb", chunk_rows=2)
    bundle = to_logical(hard, fb_dir / "chr1.vcf.gz", tmp_path / "bundle")
    got = _run_helper(bundle, "phase_pg", tmp_path / "out.tsv")
    ref = closed_form_frequencies(bundle)
    # the EM takes whole individuals: compare where every individual is complete
    complete = (got["n_used"] == 4).to_numpy() & (ref["n_used"] == 4).to_numpy() & (ref["n_haplotypes"] == 8).to_numpy()
    assert complete.sum() >= 3
    for anc in ("EUR", "AFR"):
        np.testing.assert_allclose(got.loc[complete, f"{anc}_freq"], ref.loc[complete, f"{anc}_freq"], atol=1e-5)
