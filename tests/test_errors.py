import pytest
import rfmix_reader.errors as errors

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
