# tests/test_read_flare.py
import gzip
import cudf
import pytest
import pandas as pd
import dask.array as da
import rfmix_reader.readers.read_flare as flare


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


def test_read_flare(tmp_flare_dir, monkeypatch):
    def fake_get_prefixes(prefix, mode, verbose):
        return [{
            "anc.vcf.gz": str(tmp_flare_dir / "chr21.anc.vcf"),
            "global.anc.gz": str(tmp_flare_dir / "chr21.global.anc.gz"),
        }]
    monkeypatch.setattr(flare, "get_prefixes", fake_get_prefixes)

    loci_df, g_anc, local_array = flare.read_flare(str(tmp_flare_dir))
    assert isinstance(loci_df, (pd.DataFrame, cudf.DataFrame))
    assert isinstance(g_anc, (pd.DataFrame, cudf.DataFrame))
    assert isinstance(local_array, da.Array)
    assert loci_df.shape[0] == 2
    assert g_anc.shape[0] == 2
    assert local_array.shape[2] == 2
