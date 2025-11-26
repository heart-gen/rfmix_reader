import pytest

pytest.importorskip("tqdm")

from rfmix_reader.io.errors import BinaryFileNotFoundError

# ---------------------------
# ERRORHANDLING TESTS
# ---------------------------

def test_binary_file_not_found_error_message():
    with pytest.raises(BinaryFileNotFoundError) as excinfo:
        raise BinaryFileNotFoundError("test.fb", "/tmp/bin")
    msg = str(excinfo.value)
    assert "Binary File Not Found" in msg
    assert "test.fb" in msg
    assert "/tmp/bin" in msg
