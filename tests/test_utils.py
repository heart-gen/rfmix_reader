import pytest
from pathlib import Path

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import rfmix_reader.utils as utils


def test__read_file_with_and_without_pbar(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / f"f{i}.txt"
        f.write_text(str(i))
        files.append(str(f))

    out = utils._read_file(files, lambda fn: Path(fn).read_text())
    assert out == ["0", "1", "2"]

    # With tqdm pbar
    class DummyPbar:
        def __init__(self): self.count = 0
        def update(self, n): self.count += n
    pbar = DummyPbar()
    out = utils._read_file(files, lambda fn: Path(fn).read_text(), pbar=pbar)
    assert pbar.count == len(files)


def test__clean_prefixes_basic():
    prefixes = ["/tmp/chr1.fb.tsv", "/tmp/chr2.rfmix.Q", "/tmp/ignore.logs"]
    out = utils._clean_prefixes(prefixes)
    assert any("chr1" in x for x in out)
    assert all(not x.endswith(".logs") for x in out)


def test_get_prefixes_rfmix_and_flare(tmp_path):
    # Create fake files
    f1 = tmp_path / "chr1.fb.tsv"
    f1.write_text("dummy")
    f2 = tmp_path / "chr1.rfmix.Q"
    f2.write_text("dummy")

    out = utils.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert isinstance(out, list)
    assert out and "fb.tsv" in list(out[0].keys())[0]

    # Flare mode: expect empty because no anc/global files
    with pytest.raises(FileNotFoundError):
        utils.get_prefixes(str(tmp_path), mode="flare")


def test__text_to_binary_and_process_file(tmp_path):
    fn = tmp_path / "test.fb.tsv"
    # Write a TSV with 2 header lines and numeric data
    with open(fn, "w") as f:
        f.write("header\nheader2\n")
        f.write("a b c d 1.0 2.0\n")
        f.write("e f g h 3.0 4.0\n")

    outbin = tmp_path / "test.bin"
    utils._text_to_binary(str(fn), str(outbin))
    arr = np.fromfile(outbin, dtype=np.float32).reshape(2, 2)
    assert np.allclose(arr, [[1, 2], [3, 4]])

    # Also test _process_file wrapper
    utils._process_file((str(fn), str(tmp_path)))
    assert (tmp_path / "test.bin").exists()


def test__generate_binary_files(tmp_path):
    fn = tmp_path / "t.fb.tsv"
    with open(fn, "w") as f:
        f.write("h\nh\n")
        f.write("a b c d 5.0 6.0\n")

    utils._generate_binary_files([str(fn)], str(tmp_path))
    assert (tmp_path / "t.bin").exists()


def test_delete_files_or_directories(tmp_path):
    f = tmp_path / "deleteme.txt"
    f.write_text("x")
    utils.delete_files_or_directories([str(f)])
    assert not f.exists()


def test_get_pops_and_sample_names():
    df = pd.DataFrame({
        "sample_id": ["S1", "S2"],
        "chrom": ["chr1", "chr1"],
        "AFR": [0.1, 0.2],
        "EUR": [0.9, 0.8],
    })
    pops = utils.get_pops(df)
    assert "AFR" in pops and "EUR" in pops
    samples = utils.get_sample_names(df)
    assert set(samples) == {"S1", "S2"}


def test_create_binaries_wraps(tmp_path, monkeypatch):
    # Create a dummy fb file so get_prefixes works
    fbfile = tmp_path / "chr1.fb.tsv"
    fbfile.write_text("h\nh\n a b c d 1.0\n")

    monkeypatch.setattr(utils, "_generate_binary_files", lambda fb, bd: None)
    utils.create_binaries(str(tmp_path), str(tmp_path / "out"))
    assert (tmp_path / "out").exists()
