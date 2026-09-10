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
    admix = np.zeros((1,1,1))
    out = viz.generate_tagore_bed(df, g_anc, admix, 0)
    assert "#chr" in out.columns


def test_generate_tagore_bed_end_to_end(msp_dir):
    from rfmix_reader.readers.read_msp import read_rfmix

    loci, g_anc, admix = read_rfmix(str(msp_dir), verbose=False)
    out = viz.generate_tagore_bed(loci, g_anc, admix, 0, min_segment=1, verbose=False)
    for col in ("#chr", "start", "stop", "feature", "size", "color", "chrCopy"):
        assert col in out.columns
    # every row of the BED contributes allele_count rows; total = sum of counts
    from rfmix_reader.io.loci_bed import admix_to_bed_individual
    bed = admix_to_bed_individual(loci, g_anc, admix, 0, min_segment=1, verbose=False)
    assert out.shape[0] == int(bed[["Sample_1_EUR", "Sample_1_AFR"]].to_numpy().sum())
    assert set(out["chrCopy"]) <= {1, 2}


def test_save_multi_format(tmp_path):
    fn = tmp_path / "fig"
    plt.figure()
    viz.save_multi_format(str(fn), formats=("png",))
    assert (tmp_path / "fig.png").exists()

