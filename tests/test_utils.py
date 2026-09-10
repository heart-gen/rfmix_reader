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


@pytest.mark.parametrize("chrom", ["chrX", "chrY", "chrM"])
def test_clean_prefixes_sex_chromosomes(chrom):
    """_clean_prefixes must not silently drop sex/mitochondrial chromosomes (Bug 3)."""
    prefixes = [f"/tmp/{chrom}.fb.tsv", f"/tmp/{chrom}.rfmix.Q"]
    out = utils._clean_prefixes(prefixes)
    assert out, f"_clean_prefixes returned empty for {chrom} — sex chromosomes are being dropped"
    assert any(chrom.lower() in x.lower() for x in out), (
        f"Expected {chrom} in cleaned prefixes, got: {out}"
    )


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

    monkeypatch.setattr(utils, "_generate_binary_files", lambda fb, bd, **kw: None)
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
    monkeypatch.setattr(utils, "_generate_binary_files", lambda fb, bd, **kw: None)

    with pytest.raises(RuntimeError, match="Both compressed and uncompressed"):
        utils.create_binaries(str(tmp_path), str(tmp_path / "out"))


def test_create_binaries_missing_prefix_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.create_binaries(str(tmp_path / "nothing_here"), str(tmp_path / "out"))


def test_create_binaries_chrom_filter(tmp_path, monkeypatch):
    for c in ("chr1", "chr2"):
        (tmp_path / f"{c}.fb.tsv").write_text("h\nh\n")
    seen = {}
    monkeypatch.setattr(utils, "_generate_binary_files",
                        lambda fb, bd, **kw: seen.setdefault("files", fb))
    utils.create_binaries(str(tmp_path), str(tmp_path / "out"), chrom="2", verbose=False)
    assert [str(f).endswith("chr2.fb.tsv") for f in seen["files"]] == [True]


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def _touch(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_text("x")


def test_get_prefixes_prefix_mode(tmp_path):
    _touch(tmp_path, "run_chr1.fb.tsv", "run_chr1.rfmix.Q", "run_chr2.fb.tsv", "other_chr3.fb.tsv")
    out = utils.get_prefixes(str(tmp_path / "run_"), mode="rfmix", verbose=False)
    assert [Path(m["fb.tsv"]).name for m in out] == ["run_chr1.fb.tsv", "run_chr2.fb.tsv"]
    assert "rfmix.Q" in out[0] and "rfmix.Q" not in out[1]

    one = utils.get_prefixes(str(tmp_path / "run_chr1"), mode="rfmix", verbose=False)
    assert len(one) == 1 and set(one[0]) == {"fb.tsv", "rfmix.Q"}

    single_file = utils.get_prefixes(str(tmp_path / "run_chr2.fb.tsv"), mode="rfmix", verbose=False)
    assert len(single_file) == 1


def test_get_prefixes_dotted_and_no_chr_names(tmp_path):
    _touch(tmp_path, "cohort.v2_chr1.fb.tsv", "xyz.fb.tsv", "abc.logs", "chr1.fb.tsv.tbi")
    out = utils.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert sorted(Path(m["fb.tsv"]).name for m in out) == ["cohort.v2_chr1.fb.tsv", "xyz.fb.tsv"]


def test_get_prefixes_numeric_chromosome_order(tmp_path):
    _touch(tmp_path, "chr10.fb.tsv", "chr2.fb.tsv", "chr1.fb.tsv", "chrX.fb.tsv")
    out = utils.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert [Path(m["fb.tsv"]).name for m in out] == [
        "chr1.fb.tsv", "chr2.fb.tsv", "chr10.fb.tsv", "chrX.fb.tsv"]


def test_get_prefixes_prefers_plain_over_gz(tmp_path):
    _touch(tmp_path, "chr1.fb.tsv", "chr1.fb.tsv.gz", "chr2.fb.tsv.gz")
    out = utils.get_prefixes(str(tmp_path), mode="rfmix", verbose=False)
    assert Path(out[0]["fb.tsv"]).name == "chr1.fb.tsv"
    assert Path(out[1]["fb.tsv"]).name == "chr2.fb.tsv.gz"


def test_get_prefixes_msp_mode_requires_msp(tmp_path):
    _touch(tmp_path, "chr1.rfmix.Q", "chr1.fb.tsv")
    with pytest.raises(FileNotFoundError):
        utils.get_prefixes(str(tmp_path), mode="msp", verbose=False)
    _touch(tmp_path, "chr1.msp.tsv")
    out = utils.get_prefixes(str(tmp_path), mode="msp", verbose=False)
    assert set(out[0]) == {"msp.tsv", "rfmix.Q"}


def test_get_prefixes_invalid_mode(tmp_path):
    with pytest.raises(ValueError):
        utils.get_prefixes(str(tmp_path), mode="nope")


def test_clean_prefixes_dotted_names():
    out = utils._clean_prefixes(["/x/cohort.v2_chr1.fb.tsv", "/x/cohort.v2_chr1.rfmix.Q"])
    assert out == ["/x/cohort.v2_chr1"]


def test_delete_files_or_directories_dir(tmp_path):
    d = tmp_path / "d"; d.mkdir(); (d / "f").write_text("x")
    utils.delete_files_or_directories([str(d)])
    assert not d.exists()


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
