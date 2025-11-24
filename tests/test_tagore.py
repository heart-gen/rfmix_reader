import pickle
import pytest, io
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

import rfmix_reader.tagore as tagore

def test__printif(capsys):
    tagore._printif("hi", verbose=True)
    out = capsys.readouterr().out
    assert "hi" in out
    tagore._printif("no", verbose=False)  # should print nothing


def make_bed_df():
    return pd.DataFrame({
        "#chr": ["1"],
        "start": [0],
        "stop": [100],
        "feature": [0],
        "size": [0.5],
        "color": ["#ff0000"],
        "chrCopy": [1],
    })


def test__draw_local_ancestry_and_plot_tagore(tmp_path, monkeypatch):
    bed_df = make_bed_df()
    prefix = str(tmp_path / "out")
    header, footer = "<svg>", "</svg>"

    # Should create SVG
    tagore._draw_local_ancestry(bed_df, prefix, "hg37", header, footer, verbose=False)
    assert Path(prefix + ".svg").exists()

    # Monkeypatch open_binary to return a fake pickle containing (header, footer)
    data = pickle.dumps(("<svg>", "</svg>"))
    monkeypatch.setattr(tagore, "open_binary", lambda *a, **k: io.BytesIO(data))

    # Monkeypatch svg2png/pdf to create dummy files
    def fake_svg2png(url, write_to, **k):
        Path(write_to).touch()
    def fake_svg2pdf(url, write_to, **k):
        Path(write_to).touch()
    monkeypatch.setattr(tagore, "svg2png", fake_svg2png)
    monkeypatch.setattr(tagore, "svg2pdf", fake_svg2pdf)

    tagore.plot_local_ancestry_tagore(bed_df, str(tmp_path / "fig"), "hg37", "png", force=True)
    assert (tmp_path / "fig.png").exists() or (tmp_path / "fig.pdf").exists()


def test_plot_local_ancestry_tagore_invalid_build():
    bed_df = make_bed_df()
    with pytest.raises(ValueError):
        tagore.plot_local_ancestry_tagore(bed_df, "xx", "hg39", "png")


def test__draw_local_ancestry_invalid_columns(tmp_path):
    bad_df = pd.DataFrame({"chrom": [1]})
    with pytest.raises(ValueError):
        tagore._draw_local_ancestry(bad_df, str(tmp_path / "bad"), "hg37", "<svg>", "</svg>")
