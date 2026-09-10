import numpy as np
import pytest

import rfmix_reader.readers.fb_read as fb


def _write(tmp_path, data, name="test.fb"):
    path = tmp_path / name
    data.tofile(path)
    return str(path)


@pytest.mark.parametrize("row_chunk", [1, 2, 6])
@pytest.mark.parametrize("col_chunk", [1, 3, 4, 8])
def test_read_fb_column_chunks_roundtrip(tmp_path, row_chunk, col_chunk):
    """Every (row_chunk, col_chunk) combination must reproduce the file exactly."""
    rng = np.random.default_rng(0)
    data = rng.random((6, 8), dtype=np.float32)  # non-integral posteriors
    path = _write(tmp_path, data)

    arr = fb.read_fb(path, 6, 8, row_chunk=row_chunk, col_chunk=col_chunk)
    assert arr.dtype == np.float32
    np.testing.assert_array_equal(arr.compute(), data)


def test_read_fb_keeps_fractional_values(tmp_path):
    data = np.array([[0.99, 0.01, 0.6, 0.4]], dtype=np.float32)
    path = _write(tmp_path, data)
    out = fb.read_fb(path, 1, 4, 1, 4).compute()
    np.testing.assert_allclose(out, data)


def test__read_chunk_direct(tmp_path):
    nrows, ncols = 3, 3
    data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols) + 0.5
    path = _write(tmp_path, data, "test_chunk.fb")

    out = fb._read_chunk(path, nrows, ncols, row_start=1, row_end=3,
                         col_start=1, col_end=3)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, data[1:3, 1:3])


def test__read_chunk_bounds(tmp_path):
    path = _write(tmp_path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        fb._read_chunk(path, 2, 2, 0, 3, 0, 2)
    with pytest.raises(ValueError):
        fb._read_chunk(path, 2, 2, 0, 2, 1, 3)


def test_read_fb_invalid_chunks(tmp_path):
    path = _write(tmp_path, np.zeros(4, dtype=np.float32), "dummy.fb")
    with pytest.raises(ValueError):
        fb.read_fb(path, 2, 2, row_chunk=0, col_chunk=2)
    with pytest.raises(ValueError):
        fb.read_fb(path, 2, 2, row_chunk=2, col_chunk=0)


def test_read_fb_size_mismatch_raises(tmp_path):
    path = _write(tmp_path, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="stale"):
        fb.read_fb(path, 2, 4, 2, 4)


def test_read_fb_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        fb.read_fb(str(tmp_path / "nope.bin"), 1, 1, 1, 1)
