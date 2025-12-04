import sys
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


def test_filter_paths_by_chrom_and_errors(tmp_path):
    files = [tmp_path / "chr1.data", tmp_path / "chr2.data", tmp_path / "notes.txt"]
    for f in files:
        f.write_text("x")

    matched = utils.filter_paths_by_chrom([str(f) for f in files], "chr1")
    assert matched == [str(tmp_path / "chr1.data")]

    with pytest.raises(FileNotFoundError):
        utils.filter_paths_by_chrom([str(tmp_path / "notes.txt")], "chr5")


def test_filter_file_maps_by_chrom_and_missing():
    file_maps = [
        {"fb.tsv": "/data/run_chr1.fb.tsv"},
        {"fb.tsv": "/data/run_chr2.fb.tsv"},
        {"fb.tsv": "/data/nochrom.fb.tsv"},
    ]

    filtered = utils.filter_file_maps_by_chrom(file_maps, "1", kind="test")
    assert filtered == [{"fb.tsv": "/data/run_chr1.fb.tsv"}]

    with pytest.raises(FileNotFoundError):
        utils.filter_file_maps_by_chrom(file_maps, "22", kind="test")


def test_normalize_and_extract_chrom_helpers():
    assert utils._normalize_chrom_label("ChrX") == "x"
    assert utils._normalize_chrom_label("12") == "12"

    assert utils._extract_chrom_from_path("/tmp/sample_chr10.fb.tsv") == "10"
    assert utils._extract_chrom_from_path("/tmp/run_12.fb.tsv") == "12"
    assert utils._extract_chrom_from_path("/tmp/misc.txt") is None


def test_create_binaries_conflicting_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "chr1.fb.tsv").write_text("h\nh\n")
    (tmp_path / "chr1.fb.tsv.gz").write_text("gz")

    # Avoid calling the expensive converter
    monkeypatch.setattr(utils, "_generate_binary_files", lambda fb, bd: None)

    utils.create_binaries(str(tmp_path), str(tmp_path / "out"))
    captured = capsys.readouterr().out
    assert "Both compressed and uncompressed FB files" in captured


def test_set_gpu_environment_monkeypatched(monkeypatch, capsys):
    class DummyProps:
        name = "Dummy GPU"
        total_memory = 4 * 1024 ** 3
        major, minor = 1, 0

    class DummyCuda:
        def __init__(self, count):
            self._count = count

        def device_count(self):
            return self._count

        def get_device_properties(self, idx):
            assert idx == 0
            return DummyProps()

    dummy = DummyCuda(count=1)

    monkeypatch.setitem(sys.modules, "torch", type("Mod", (), {})())
    monkeypatch.setitem(sys.modules, "torch.cuda", dummy)

    utils.set_gpu_environment()
    out = capsys.readouterr().out
    assert "Dummy GPU" in out
