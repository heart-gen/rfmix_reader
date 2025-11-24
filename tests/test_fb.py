import pytest
import numpy as np
import rfmix_reader.readers.fb_read as fb

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
    assert result.dtype == np.int32


def test__read_chunk_direct(tmp_path):
    nrows, ncols = 3, 3
    data = np.arange(nrows * ncols, dtype=np.float32).reshape(nrows, ncols)
    file_path = tmp_path / "test_chunk.fb"
    data.tofile(file_path)

    out = fb._read_chunk(str(file_path), nrows, ncols,
                         row_start=1, row_end=3,
                         col_start=1, col_end=3)

    assert out.dtype == np.int32
    assert np.all(np.isin(out, data.astype(np.int32)))


def test_read_fb_invalid_chunks(tmp_path):
    file_path = tmp_path / "dummy.fb"
    np.zeros(4, dtype=np.float32).tofile(file_path)

    with pytest.raises(ValueError):
        fb.read_fb(str(file_path), 2, 2, row_chunk=0, col_chunk=2)
    with pytest.raises(ValueError):
        fb.read_fb(str(file_path), 2, 2, row_chunk=2, col_chunk=0)
