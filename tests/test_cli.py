import os
import sys
import tempfile
import numpy as np
import pytest

import rfmix_reader._cli as cli
import rfmix_reader._constants as consts
import rfmix_reader._errorhandling as errors
import rfmix_reader._fb_read as fb


# ---------------------------
# CLI TESTS
# ---------------------------

def test_cli_invokes_create_binaries(monkeypatch):
    called = {}

    def fake_create_binaries(file_path, binary_dir):
        called["args"] = (file_path, binary_dir)

    monkeypatch.setattr(cli, "create_binaries", fake_create_binaries)

    test_args = ["prog", "input_dir", "--binary_dir", "bin_out"]
    monkeypatch.setattr(sys, "argv", test_args)

    cli.main()
    assert called["args"] == ("input_dir", "bin_out")


def test_cli_version_flag(monkeypatch, capsys):
    test_args = ["prog", "--version"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "prog" in out or cli.__version__ in out


# ---------------------------
# CONSTANTS TESTS
# ---------------------------

def test_coordinates_and_sizes():
    coords = consts.COORDINATES
    sizes = consts.CHROM_SIZES

    # Coordinates contain chr1 and X
    assert "1" in coords
    assert "X" in coords
    assert "cx" in coords["1"]

    # Sizes return dicts for both assemblies
    hg37 = sizes.get_sizes("hg37")
    hg38 = sizes.get_sizes("hg38")
    assert isinstance(hg37, dict) and "1" in hg37
    assert isinstance(hg38, dict) and "22" in hg38

    # Assemblies property works
    assert set(sizes.available_assemblies) >= {"hg37", "hg38"}


# ---------------------------
# ERRORHANDLING TESTS
# ---------------------------

def test_binary_file_not_found_error_message():
    with pytest.raises(errors.BinaryFileNotFoundError) as excinfo:
        raise errors.BinaryFileNotFoundError("test.fb", "/tmp/bin")
    msg = str(excinfo.value)
    assert "Binary File Not Found" in msg
    assert "test.fb" in msg
    assert "/tmp/bin" in msg


# ---------------------------
# FB READ TESTS
# ---------------------------

def test_read_fb_and_chunk(tmp_path):
    nrows, ncols = 4, 6
    data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
    file_path = tmp_path / "test.fb"
    data.tofile(file_path)

    arr = fb.read_fb(str(file_path), nrows, ncols, row_chunk=2, col_chunk=3)
    result = arr.compute()

    assert result.shape == (nrows, ncols)
    # Because _read_chunk casts to int32
    expected = data.astype(np.int32)
    np.testing.assert_array_equal(result, expected)


def test__read_chunk_direct(tmp_path):
    nrows, ncols = 3, 3
    data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
    file_path = tmp_path / "test_chunk.fb"
    data.tofile(file_path)

    out = fb._read_chunk(str(file_path), nrows, ncols,
                         row_start=1, row_end=3,
                         col_start=1, col_end=3)

    expected = data.astype(np.int32)[1:3, 1:3]
    np.testing.assert_array_equal(out, expected)
    
    assert out.dtype == np.int32

    expected = data.astype(np.int32)[1:3, 1:3]
    np.testing.assert_array_equal(out, expected)


def test_read_fb_invalid_chunks(tmp_path):
    file_path = tmp_path / "dummy.fb"
    np.zeros(4, dtype=np.float32).tofile(file_path)

    with pytest.raises(ValueError):
        fb.read_fb(str(file_path), 2, 2, row_chunk=0, col_chunk=2)
    with pytest.raises(ValueError):
        fb.read_fb(str(file_path), 2, 2, row_chunk=2, col_chunk=0)
