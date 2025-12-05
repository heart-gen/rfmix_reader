import pytest
from pathlib import Path

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

import rfmix_reader.viz.visualization as viz

def make_ganc_df():
    return pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "chrom": ["chr1", "chr1"],
        "AFR": [0.2, 0.4],
        "EUR": [0.8, 0.6],
    })


def test_get_global_ancestry_and_plot(tmp_path):
    g_anc = make_ganc_df()
    result = viz._get_global_ancestry(g_anc)
    assert "AFR" in result.columns and "EUR" in result.columns

    # plot_global_ancestry saves to file
    out = tmp_path / "global"
    viz.plot_global_ancestry(g_anc, save_path=str(out))
    assert (tmp_path / "global.png").exists()
    assert (tmp_path / "global.pdf").exists()


def test_plot_ancestry_by_chromosome(tmp_path):
    g_anc = make_ganc_df()
    out = tmp_path / "chromsum"
    viz.plot_ancestry_by_chromosome(g_anc, save_path=str(out))
    assert (tmp_path / "chromsum.png").exists()
    assert (tmp_path / "chromsum.pdf").exists()


def test_expand_and_annotate_tagore(monkeypatch):
    df = pd.DataFrame({
        "chromosome": ["1", "1"],
        "start": [0, 100],
        "end": [50, 200],
        "S1_AFR": [1, 0],
        "S1_EUR": [0, 1],
    })
    pops = ["AFR", "EUR"]

    expanded = viz._expand_dataframe(df, ["S1_AFR", "S1_EUR"])
    assert "sample_name" in expanded.columns

    # Force NumPy backend for chrCopy
    monkeypatch.setattr(viz, "cp", np)
    ann = viz._annotate_tagore(df, ["S1_AFR", "S1_EUR"], pops)
    assert "#chr" in ann.columns and "chrCopy" in ann.columns


def test_generate_tagore_bed(monkeypatch):
    df = pd.DataFrame({
        "chromosome": ["1"],
        "start": [0],
        "end": [100],
        "S1_AFR": [1],
    })
    g_anc = pd.DataFrame({
        "sample_id": ["S1"], "chrom": ["1"], "AFR": [1.0]
    })
    monkeypatch.setattr(viz, "admix_to_bed_individual",
                        lambda loci, g, a, sn, cs, ms, v: df)
    # Force NumPy backend
    monkeypatch.setattr(viz, "cp", np)
    admix = np.zeros((1,1,1))
    out = viz.generate_tagore_bed(df, g_anc, admix, 0)
    assert "#chr" in out.columns


def test_save_multi_format(tmp_path):
    fn = tmp_path / "fig"
    plt.figure()
    viz.save_multi_format(str(fn), formats=("png",))
    assert (tmp_path / "fig.png").exists()

