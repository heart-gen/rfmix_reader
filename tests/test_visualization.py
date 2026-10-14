import pytest

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

    expanded = viz._expand_dataframe(df, ["S1_AFR", "S1_EUR"], pops)
    assert "sample_name" in expanded.columns
    assert expanded["sample_name"].tolist() == ["AFR", "EUR"]

    ann = viz._annotate_tagore(df, ["S1_AFR", "S1_EUR"], pops)
    assert "#chr" in ann.columns and "chrCopy" in ann.columns
    assert ann["color"].notna().all()


def test_expand_dataframe_lowercase_and_digit_pops():
    """Population labels are matched literally, not with [A-Z]+."""
    df = pd.DataFrame({
        "chromosome": ["1"], "start": [0], "end": [50],
        "S_1_pop1": [1], "S_1_Yri2": [1],
    })
    expanded = viz._expand_dataframe(df, ["S_1_pop1", "S_1_Yri2"], ["pop1", "Yri2"])
    assert sorted(expanded["sample_name"]) == ["Yri2", "pop1"]


def test_save_multi_format(tmp_path):
    fn = tmp_path / "fig"
    plt.figure()
    viz.save_multi_format(str(fn), formats=("png",))
    assert (tmp_path / "fig.png").exists()



def test_plot_global_ancestry_legend_matches_columns(monkeypatch):
    """Each legend entry is drawn from the ancestry column of the same name,
    averaged over chromosomes, regardless of column order (issue #32)."""
    g_anc = pd.DataFrame({
        "sample_id": ["S1", "S2", "S1", "S2"],
        "EUR": [0.1, 0.7, 0.3, 0.9],
        "AFR": [0.9, 0.3, 0.7, 0.1],
        "chrom": ["chr1", "chr1", "chr2", "chr2"],
    })
    expected = {"EUR": [0.2, 0.8], "AFR": [0.8, 0.2]}
    captured = {}

    def fake_show():
        ax = plt.gcf().axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        for label, container in zip(labels, ax.containers):
            captured[label] = [bar.get_height() for bar in container]
    monkeypatch.setattr(plt, "show", fake_show)

    viz.plot_global_ancestry(g_anc, save_path=None)
    assert list(captured) == ["EUR", "AFR"]
    for label, heights in expected.items():
        np.testing.assert_allclose(captured[label], heights)

    captured.clear()
    viz.plot_global_ancestry(g_anc[["sample_id", "AFR", "EUR", "chrom"]], save_path=None)
    assert list(captured) == ["AFR", "EUR"]
    for label, heights in expected.items():
        np.testing.assert_allclose(captured[label], heights)
