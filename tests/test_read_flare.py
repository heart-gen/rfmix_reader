import gzip
import importlib
import shutil

import numpy as np
import pytest
import dask.array as da

from rfmix_reader.utils import get_pops

flare = importlib.import_module("rfmix_reader.readers.read_flare")

# tests/data/flare: ##ANCESTRY=<EUR=0,AFR=1>; axis 2 == [EUR, AFR]
EXPECTED = np.array([
    [[2, 0], [1, 1]],     # rs1: S1 0/0, S2 0/1
    [[1, 1], [2, 0]],     # rs2: S1 1/0, S2 0/0
    [[0, 2], [-1, -1]],   # rs3: S1 1/1, S2 ./1 -> missing
    [[1, 1], [0, 2]],     # rs4
], dtype=np.int8)


@pytest.fixture
def plain_vcf(tmp_path):
    """Uncompressed FLARE-style VCF (cyvcf2 iterates it without an index)."""
    vcf_content = """##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">
##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">
##ANCESTRY=<EUR=0,AFR=1>
##contig=<ID=chr21>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1\tSample_2
chr21\t5030578\trs1\tC\tT\t.\tPASS\t.\tGT:AN1:AN2\t0|0:0:0\t0|1:0:1
chr21\t5030588\trs2\tT\tC\t.\tPASS\t.\tGT:AN1:AN2\t0|0:1:0\t0|0:0:0
"""
    path = tmp_path / "chr21.anc.vcf"
    path.write_text(vcf_content)
    return path


def test_parse_ancestry_header(flare_dir):
    mapping = flare._parse_ancestry_header(str(flare_dir / "chr21.anc.vcf.gz"))
    assert mapping == {"EUR": 0, "AFR": 1}


def test_parse_ancestry_header_missing(tmp_path):
    path = tmp_path / "x.vcf"
    path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    with pytest.raises(ValueError, match="ANCESTRY"):
        flare._parse_ancestry_header(str(path))


def test_load_vcf_info_and_loci(plain_vcf):
    chunks = list(flare._load_vcf_info(str(plain_vcf), chunk_size=1))
    assert len(chunks) == 2
    assert list(chunks[0].columns) == ["chromosome", "physical_position"]

    df = flare._read_loci(str(plain_vcf), chunk_size=1)
    assert df["i"].tolist() == [0, 1]
    assert df["physical_position"].dtype == np.int32


def test_read_anc(flare_dir):
    df = flare._read_anc(str(flare_dir / "chr21.global.anc.gz"))
    assert list(df.columns) == ["sample_id", "EUR", "AFR", "chrom"]
    assert df["chrom"].iloc[0] == "chr21"
    assert df["EUR"].dtype == np.float32


def test_load_haplotypes_header_order_and_sentinel(flare_dir):
    arr = flare._load_haplotypes(str(flare_dir / "chr21.anc.vcf.gz"), chunk_size=3)
    assert isinstance(arr, da.Array)
    assert arr.dtype == np.int8
    assert arr.shape == (4, 2, 2)
    assert arr.chunks[0] == (3, 1)
    np.testing.assert_array_equal(arr.compute(), EXPECTED)


def test_load_haplotypes_plain_vcf(plain_vcf):
    result = flare._load_haplotypes(str(plain_vcf), chunk_size=10).compute()
    EUR, AFR = 0, 1
    assert result[0, 0, EUR] == 2 and result[0, 0, AFR] == 0
    assert result[0, 1, EUR] == 1 and result[0, 1, AFR] == 1
    assert result[1, 0, EUR] == 1 and result[1, 0, AFR] == 1
    assert result[1, 1, EUR] == 2 and result[1, 1, AFR] == 0


def test_diploid_counts_from_haps_missing_codes_are_sentinel():
    an1 = np.array([-2147483648, -1, 0, 1, 2], dtype=np.int32)
    an2 = np.array([0, 1, -1, 1, 0], dtype=np.int32)
    result = flare._diploid_counts_from_haps(an1, an2, 2)
    assert result.dtype == np.int8
    assert (result[[0, 1, 2, 4], :] == -1).all()
    np.testing.assert_array_equal(result[3, :], [0, 2])


def test_read_flare_end_to_end(flare_dir):
    loci_df, g_anc, local_array = flare.read_flare(str(flare_dir), verbose=False)
    assert loci_df.shape[0] == 4
    assert loci_df["i"].tolist() == [0, 1, 2, 3]
    assert list(get_pops(g_anc)) == ["EUR", "AFR"]
    assert local_array.dtype == np.int8
    np.testing.assert_array_equal(local_array.compute(), EXPECTED)


def test_read_flare_g_anc_reordered_to_header(flare_dir, tmp_path):
    """global.anc columns not in ##ANCESTRY order are realigned to axis 2."""
    for name in ("chr21.anc.vcf.gz", "chr21.anc.vcf.gz.tbi"):
        shutil.copy(flare_dir / name, tmp_path / name)
    with gzip.open(tmp_path / "chr21.global.anc.gz", "wt") as fh:
        fh.write("SAMPLE\tAFR\tEUR\nSample_1\t0.375\t0.625\nSample_2\t0.625\t0.375\n")

    _, g_anc, _ = flare.read_flare(str(tmp_path), verbose=False)
    assert list(g_anc.columns) == ["sample_id", "EUR", "AFR", "chrom"]
    assert g_anc.loc[0, "EUR"] == pytest.approx(0.625)


def test_read_flare_chrom_filter(flare_dir):
    _, _, arr = flare.read_flare(str(flare_dir), verbose=False, chrom="21")
    assert arr.shape[0] == 4
    with pytest.raises(FileNotFoundError):
        flare.read_flare(str(flare_dir), verbose=False, chrom="1")
