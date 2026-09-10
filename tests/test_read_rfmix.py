"""
Tests for the ``.fb.tsv`` reader (``read_rfmix_fb``) on a small committed
fixture with fractional posteriors.  Runs on CPU; no cuDF required.
"""
import importlib

import numpy as np
import pandas as pd
import pytest
import dask.array as da

from rfmix_reader.io import Chunk
from rfmix_reader.utils import create_binaries, get_pops

rfmix = importlib.import_module("rfmix_reader.readers.read_rfmix")


def _text_matrix(fb_dir):
    """Posteriors of the fixture parsed independently with pandas."""
    df = pd.read_csv(fb_dir / "chr1.fb.tsv", sep="\t", skiprows=1, header=0)
    return df.iloc[:, 4:].to_numpy(dtype=np.float32), df


def _expected_counts(X, npops):
    b4 = X.reshape(X.shape[0], -1, 2, npops)
    codes = b4.argmax(-1)
    eye = np.eye(npops, dtype=np.int8)
    out = eye[codes[..., 0]] + eye[codes[..., 1]]
    missing = ~(b4 > 0).any(-1)
    out[missing.any(-1)] = -1
    return out


@pytest.fixture(scope="module")
def fb_binaries(fb_dir, tmp_path_factory):
    bin_dir = tmp_path_factory.mktemp("bin")
    create_binaries(str(fb_dir), str(bin_dir), verbose=False)
    return bin_dir


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test__read_fb_pops_header(fb_dir):
    assert rfmix._read_fb_pops(str(fb_dir / "chr1.fb.tsv")) == ["EUR", "AFR"]


def test__read_fb_pops_bad_header(tmp_path):
    fn = tmp_path / "chr1.fb.tsv"
    fn.write_text("chromosome\tphysical_position\n")
    with pytest.raises(ValueError, match="reference_panel_population"):
        rfmix._read_fb_pops(str(fn))


def test__read_tsv_and_loci(fb_dir):
    fn = str(fb_dir / "chr1.fb.tsv")
    df = rfmix._read_tsv(fn)
    assert list(df.columns) == ["chromosome", "physical_position"]
    assert df.shape[0] == 5

    loci = rfmix._read_loci(fn)
    assert loci["i"].tolist() == [0, 1, 2, 3, 4]


def test__read_tsv_file_not_found():
    with pytest.raises(FileNotFoundError):
        rfmix._read_tsv("nonexistent.tsv")


def test__read_Q_and_Qnoi(fb_dir):
    fn = str(fb_dir / "chr1.rfmix.Q")
    out = rfmix._read_Q(fn)
    assert list(out.columns) == ["sample_id", "EUR", "AFR", "chrom"]
    assert out["chrom"].iloc[0] == "chr1"
    assert out["EUR"].dtype == np.float32

    out2 = rfmix._read_Q_noi(fn)
    assert "chrom" not in out2.columns
    assert out2["sample_id"].tolist() == ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]


# ---------------------------------------------------------------------------
# posteriors -> hard counts
# ---------------------------------------------------------------------------

def test_posteriors_to_counts_argmax():
    X = da.from_array(np.array([
        [0.99, 0.01, 0.40, 0.60],   # hap1 -> pop0, hap2 -> pop1
        [0.90, 0.10, 0.80, 0.20],   # both pop0
        [0.00, 0.00, 0.70, 0.30],   # hap1 has no mass -> missing
    ], dtype=np.float32), chunks=(2, 4))
    out = rfmix._posteriors_to_counts(X, npops=2)
    assert isinstance(out, da.Array)
    assert out.dtype == np.int8
    assert out.shape == (3, 1, 2)
    np.testing.assert_array_equal(out.compute()[:, 0, :], [[1, 1], [2, 0], [-1, -1]])


def test_posteriors_to_counts_misaligned_chunks_are_fixed():
    X = da.from_array(np.zeros((2, 8), dtype=np.float32) + 0.5, chunks=(2, 3))
    out = rfmix._posteriors_to_counts(X, npops=2)
    assert out.shape == (2, 2, 2)
    out.compute()


@pytest.mark.parametrize("ncols", [5, 6])
def test_posteriors_to_counts_invalid_columns(ncols):
    X = da.from_array(np.ones((2, ncols), dtype=np.float32))
    with pytest.raises(ValueError, match="divisible"):
        rfmix._posteriors_to_counts(X, npops=2)


# ---------------------------------------------------------------------------
# _read_fb through narrow column chunks (A0 + A1 together)
# ---------------------------------------------------------------------------

def test__read_fb_missing_binary(tmp_path, fb_dir):
    with pytest.raises(rfmix.BinaryFileNotFoundError):
        rfmix._read_fb(str(fb_dir / "chr1.fb.tsv"), 4, 5, ["EUR", "AFR"],
                       str(tmp_path), Chunk())


def test__read_fb_narrow_column_chunks(fb_dir, fb_binaries):
    X_text, _ = _text_matrix(fb_dir)
    admix, X = rfmix._read_fb(
        str(fb_dir / "chr1.fb.tsv"), 4, 5, ["EUR", "AFR"], str(fb_binaries),
        Chunk(nsamples=1, nloci=2),
    )
    assert X.dtype == np.float32
    assert X.chunks[1] == (4, 4, 4, 4)          # one sample per column block
    np.testing.assert_allclose(X.compute(), X_text, atol=1e-6)
    np.testing.assert_array_equal(admix.compute(), _expected_counts(X_text, 2))


def test__read_fb_default_chunk_is_whole_samples(fb_dir, fb_binaries):
    admix, X = rfmix._read_fb(
        str(fb_dir / "chr1.fb.tsv"), 4, 5, ["EUR", "AFR"], str(fb_binaries), None,
    )
    assert all(c % 4 == 0 for c in X.chunks[1])


# ---------------------------------------------------------------------------
# read_rfmix_fb end to end
# ---------------------------------------------------------------------------

def test_read_rfmix_fb_end_to_end(fb_dir, fb_binaries):
    loci_df, g_anc, admix, X_raw = rfmix.read_rfmix_fb(
        str(fb_dir), binary_dir=str(fb_binaries), verbose=False,
        return_original=True, chunk=Chunk(nsamples=1, nloci=2),
    )
    X_text, text_df = _text_matrix(fb_dir)

    assert isinstance(admix, da.Array) and admix.dtype == np.int8
    assert admix.shape == (5, 4, 2)
    assert X_raw.dtype == np.float32
    np.testing.assert_allclose(X_raw.compute(), X_text, atol=1e-6)
    np.testing.assert_array_equal(admix.compute(), _expected_counts(X_text, 2))

    # axis 2 == reference-panel order == g_anc columns
    assert list(get_pops(g_anc)) == ["EUR", "AFR"]
    assert loci_df["physical_position"].tolist() == text_df["physical_position"].tolist()
    assert loci_df["i"].tolist() == list(range(5))

    counts = admix.compute()
    valid = counts[:, :, 0] >= 0
    assert (counts.sum(axis=2)[valid] == 2).all()
    assert (~valid).sum() == 1  # the one haplotype without posterior mass


def test_read_rfmix_fb_generate_binary_and_chrom(fb_dir, tmp_path):
    loci_df, g_anc, admix = rfmix.read_rfmix_fb(
        str(fb_dir), binary_dir=str(tmp_path / "bin"), generate_binary=True,
        verbose=False, chrom="1",
    )
    assert (tmp_path / "bin" / "chr1.bin").exists()
    assert admix.shape == (5, 4, 2)


def test_read_rfmix_fb_chrom_not_found(fb_dir, fb_binaries):
    with pytest.raises(FileNotFoundError):
        rfmix.read_rfmix_fb(str(fb_dir), binary_dir=str(fb_binaries),
                            verbose=False, chrom="22")


def test_read_rfmix_fb_q_columns_reordered(fb_dir, fb_binaries, tmp_path):
    """A .Q whose columns are not in reference-panel order is realigned."""
    import shutil
    shutil.copy(fb_dir / "chr1.fb.tsv", tmp_path / "chr1.fb.tsv")
    q = pd.read_csv(fb_dir / "chr1.rfmix.Q", sep="\t", skiprows=1)
    q = q[["#sample", "AFR", "EUR"]]
    with open(tmp_path / "chr1.rfmix.Q", "w") as fh:
        fh.write("#rfmix diploid global ancestry .Q format output\n")
        q.to_csv(fh, sep="\t", index=False)

    _, g_anc, _ = rfmix.read_rfmix_fb(str(tmp_path), binary_dir=str(fb_binaries),
                                      verbose=False)
    assert list(g_anc.columns) == ["sample_id", "EUR", "AFR", "chrom"]
