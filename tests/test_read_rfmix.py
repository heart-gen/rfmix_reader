import os
import cudf
import pytest
import numpy as np
import pandas as pd
import dask.array as da

from rfmix_reader.io import Chunk
import rfmix_reader.readers.read_rfmix as rfmix


def make_dummy_tsv(tmp_path, fname="chr21.fb.tsv"):
    """Helper to create a simple TSV file for loci/Q matrix tests."""
    fn = tmp_path / fname
    df = pd.DataFrame({
        "chromosome": ["chr21", "chr21"],
        "physical_position": [5030578, 5030588],
    })
    df.to_csv(fn, sep="\t", index=False)
    return str(fn)


def test__read_tsv_and_loci(tmp_path):
    fn = make_dummy_tsv(tmp_path)
    df = rfmix._read_tsv(fn)
    assert "chromosome" in df.columns
    assert "physical_position" in df.columns

    loci = rfmix._read_loci(fn)
    assert "i" in loci.columns

    i_col = loci["i"]
    if hasattr(i_col, "to_arrow"):  # cuDF
        vals = i_col.to_arrow().to_pylist()
    else:
        vals = i_col.tolist()
    assert vals == [0, 1]


def test__read_csv_and_types(tmp_path):
    fn = make_dummy_tsv(tmp_path, "chr21.Q")
    header = {"sample_id": "category", "val": np.int32}
    # Create fake file to satisfy _types
    df = pd.DataFrame({"sample_id": ["A", "B"], "val": [1, 2]})
    df.to_csv(fn, sep="\t", index=False)
    out = rfmix._read_csv(str(fn), header)
    assert isinstance(out, (pd.DataFrame, cudf.DataFrame))
    assert list(out.columns) == ["sample_id", "val"]

    header2 = rfmix._types(str(fn))
    assert isinstance(header2, dict)
    assert "sample_id" in header2


def test__read_Q_and_Qnoi(tmp_path):
    fn = make_dummy_tsv(tmp_path, "chr21.Q")
    df = pd.DataFrame({"sample_id": ["S1"], "A": [0.5], "B": [0.5]})
    df.to_csv(fn, sep="\t", index=False)

    out = rfmix._read_Q(str(fn))
    assert "chrom" in out.columns
    assert out["chrom"].iloc[0] == "chr21"

    out2 = rfmix._read_Q_noi(str(fn))
    assert "sample_id" in out2.columns


def test__subset_populations_valid():
    X = da.from_array(np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ]))
    # 2 populations → expect shape (2,1,2)
    out = rfmix._subset_populations(X, npops=2)
    assert isinstance(out, da.Array)
    assert out.shape == (2, 1, 2)


def test__subset_populations_invalid_columns():
    X = da.from_array(np.ones((2, 5)))
    with pytest.raises(ValueError, match="divisible"):
        rfmix._subset_populations(X, npops=2)


def test__subset_populations_odd_cols():
    X = da.from_array(np.ones((2, 6)))
    with pytest.raises(ValueError, match="even"):
        rfmix._subset_populations(X, npops=2).compute()


def test__read_fb_missing_binary(tmp_path):
    fn = make_dummy_tsv(tmp_path, "chr21.fb.tsv")
    nsamples, nloci, pops = 2, 2, ["A", "B"]
    with pytest.raises(rfmix.BinaryFileNotFoundError):
        rfmix._read_fb(fn, nsamples, nloci, pops, str(tmp_path), Chunk())


def test__read_tsv_file_not_found():
    with pytest.raises(FileNotFoundError):
        rfmix._read_tsv("nonexistent.tsv")
