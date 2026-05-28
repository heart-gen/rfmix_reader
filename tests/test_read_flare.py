# tests/test_read_flare.py
import gzip
import importlib
import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
da = pytest.importorskip("dask.array")

try:
    import cudf
    _has_cudf = True
except ImportError:
    cudf = pd  # fallback so isinstance checks don't crash
    _has_cudf = False

# Use importlib to get the module, not the function re-exported by __init__.py
flare = importlib.import_module("rfmix_reader.readers.read_flare")


@pytest.fixture
def tmp_flare_dir(tmp_path):
    """
    Create a temporary directory with minimal FLARE-style outputs:
    - chr21.anc.vcf (with ancestry header + 2 variants, no bgzip/tabix)
    - chr21.global.anc.gz (simple global ancestry table)
    """
    d = tmp_path

    # Minimal FLARE-style VCF with GT, AN1, AN2
    vcf_content = """##fileformat=VCFv4.2
##filedate=20250423
##source=flare.test
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">
##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">
##ANCESTRY=<EUR=0,AFR=1>
##contig=<ID=chr21>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1\tSample_2
chr21\t5030578\trs1\tC\tT\t.\tPASS\t.\tGT:AN1:AN2\t0|0:0:0\t0|1:0:1
chr21\t5030588\trs2\tT\tC\t.\tPASS\t.\tGT:AN1:AN2\t0|0:1:0\t0|0:0:0
"""
    vcf_path = d / "chr21.anc.vcf"
    with open(vcf_path, "w") as f:
        f.write(vcf_content)

    # Global ancestry file (tab-delimited, typical of FLARE outputs)
    global_content = "SAMPLE\tEUR\tAFR\nSample_1\t0.7\t0.3\nSample_2\t0.2\t0.8\n"
    global_path = d / "chr21.global.anc.gz"
    with gzip.open(global_path, "wt") as f:
        f.write(global_content)

    return d


def test_parse_ancestry_header(tmp_flare_dir):
    vcf_file = tmp_flare_dir / "chr21.anc.vcf"
    mapping = flare._parse_ancestry_header(str(vcf_file))
    assert mapping == {"EUR": 0, "AFR": 1}


def test_load_vcf_info(tmp_flare_dir):
    vcf_file = tmp_flare_dir / "chr21.anc.vcf"
    chunks = list(flare._load_vcf_info(str(vcf_file), chunk_size=1))
    df = chunks[0]
    assert isinstance(df, (pd.DataFrame, cudf.DataFrame))  # use same import style as module
    assert "chromosome" in df.columns
    assert "physical_position" in df.columns


def test_read_loci(tmp_flare_dir):
    vcf_file = tmp_flare_dir / "chr21.anc.vcf"
    df = flare._read_loci(str(vcf_file), chunk_size=1)
    assert "i" in df.columns
    assert df.shape[0] == 2  # 2 variants


def test_read_anc(tmp_flare_dir):
    global_file = tmp_flare_dir / "chr21.global.anc.gz"
    df = flare._read_anc(str(global_file))
    assert "chrom" in df.columns
    assert set(df.columns) >= {"sample_id", "EUR", "AFR"}


def test_load_haplotypes(tmp_flare_dir):
    vcf_file = tmp_flare_dir / "chr21.anc.vcf"
    arr = flare._load_haplotypes(str(vcf_file), chunk_size=10)
    assert isinstance(arr, da.Array)
    # shape (variants, samples, ancestries)
    assert arr.shape[0] == 2  # 2 variants
    assert arr.shape[1] == 2  # 2 samples
    assert arr.shape[2] == 2  # EUR, AFR


def test_load_haplotypes_numerical_correctness(tmp_flare_dir):
    """
    Verify that _load_haplotypes produces correct ancestry count values (Bug 6).

    VCF content (after Bug 6 fix, numpy arrays are extracted before dask.delayed):
      Variant 1: Sample_1 AN1=0(EUR) AN2=0(EUR) → EUR=2, AFR=0
                 Sample_2 AN1=0(EUR) AN2=1(AFR) → EUR=1, AFR=1
      Variant 2: Sample_1 AN1=1(AFR) AN2=0(EUR) → EUR=1, AFR=1
                 Sample_2 AN1=0(EUR) AN2=0(EUR) → EUR=2, AFR=0

    ANCESTRY header: EUR=0, AFR=1  → alphabetical sort → AFR=axis0, EUR=axis1
    """
    import numpy as np

    vcf_file = tmp_flare_dir / "chr21.anc.vcf"
    arr = flare._load_haplotypes(str(vcf_file), chunk_size=10)
    result = arr.compute()  # shape (2 variants, 2 samples, 2 ancestries)

    # Ancestry axis is sorted alphabetically: AFR=0, EUR=1
    AFR, EUR = 0, 1

    # Variant 0
    # Sample_1: AN1=0(EUR), AN2=0(EUR) → AFR=0, EUR=2
    assert result[0, 0, AFR] == 0 and result[0, 0, EUR] == 2, (
        f"Variant 0 Sample_1 wrong: {result[0, 0, :]}"
    )
    # Sample_2: AN1=0(EUR), AN2=1(AFR) → AFR=1, EUR=1
    assert result[0, 1, AFR] == 1 and result[0, 1, EUR] == 1, (
        f"Variant 0 Sample_2 wrong: {result[0, 1, :]}"
    )

    # Variant 1
    # Sample_1: AN1=1(AFR), AN2=0(EUR) → AFR=1, EUR=1
    assert result[1, 0, AFR] == 1 and result[1, 0, EUR] == 1, (
        f"Variant 1 Sample_1 wrong: {result[1, 0, :]}"
    )
    # Sample_2: AN1=0(EUR), AN2=0(EUR) → AFR=0, EUR=2
    assert result[1, 1, AFR] == 0 and result[1, 1, EUR] == 2, (
        f"Variant 1 Sample_2 wrong: {result[1, 1, :]}"
    )


def test_diploid_counts_from_haps_missing_codes_are_nan():
    import numpy as np

    eye = np.eye(2, dtype=np.float32)
    an1 = np.array([-2147483648, -1, 0, 1, 2], dtype=np.int32)
    an2 = np.array([0, 1, -1, 1, 0], dtype=np.int32)

    result = flare._diploid_counts_from_haps(an1, an2, eye)

    assert np.isnan(result[[0, 1, 2, 4], :]).all()
    np.testing.assert_array_equal(result[3, :], [0, 2])


def test_read_flare(tmp_flare_dir, monkeypatch):
    def fake_get_prefixes(prefix, mode, verbose):
        return [{
            "anc.vcf": str(tmp_flare_dir / "chr21.anc.vcf"),
            "global.anc": str(tmp_flare_dir / "chr21.global.anc.gz"),
        }]
    monkeypatch.setattr(flare, "get_prefixes", fake_get_prefixes)

    loci_df, g_anc, local_array = flare.read_flare(str(tmp_flare_dir))
    assert isinstance(loci_df, (pd.DataFrame, cudf.DataFrame))
    assert isinstance(g_anc, (pd.DataFrame, cudf.DataFrame))
    assert isinstance(local_array, da.Array)
    assert loci_df.shape[0] == 2
    assert g_anc.shape[0] == 2
    assert local_array.shape[2] == 2
