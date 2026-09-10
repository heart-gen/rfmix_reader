import os
import textwrap

import numpy as np
import pytest

from rfmix_reader.readers.read_simu import (
    MISSING,
    _build_mapper,
    _codes_to_counts,
    _init_vcf,
    _map_pop_to_codes,
    _parse_pop_labels,
    read_simu,
)
from rfmix_reader.utils import get_pops

# Expected diploid counts for tests/data/simu (pops sorted: CEU, NAT, YRI)
EXPECTED = np.array([
    [[0, 0, 2], [1, 0, 1], [1, 1, 0], [0, 1, 1]],   # pos 100
    [[1, 0, 1], [1, 0, 1], [0, 2, 0], [0, 1, 1]],   # pos 5000
    [[2, 0, 0], [0, 0, 2], [1, 1, 0], [1, 1, 0]],   # pos 1500000 (2nd region)
], dtype=np.int8)


def test_parse_pop_labels_bp(simu_dir):
    assert _parse_pop_labels(str(simu_dir / "chr21.vcf.gz")) == ["CEU", "NAT", "YRI"]


def test_parse_pop_labels_fallback(tmp_path):
    """Remove .bp and ensure fallback to VCF POP works."""
    vcf_path = tmp_path / "chr21.vcf"
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##contig=<ID=chr21,length=46709983>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=POP,Number=2,Type=String,Description="Origin Population">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1
    """)
    record = "chr21\t5030578\t21:5030578:C:T\tC\tT\t.\tPASS\t.\tGT:POP\t0|0:YRI,CEU"
    vcf_path.write_text(header + record + "\n")
    labels = _parse_pop_labels(str(vcf_path))
    assert labels == ["CEU", "YRI"]


def test_map_pop_to_codes_and_counts():
    ancestries, _ = _build_mapper(["CEU", "YRI"])
    arr = np.array([["YRI,CEU", "CEU,YRI", "XXX,YRI"]])
    codes = _map_pop_to_codes(arr, ancestries)
    assert codes.shape == (1, 3, 2)
    assert codes[0, 2, 0] == MISSING
    counts = _codes_to_counts(codes, 2)
    assert counts.dtype == np.int8
    np.testing.assert_array_equal(counts[0], [[1, 1], [1, 1], [-1, -1]])


def test_init_vcf_uses_record_contig_not_first_header_contig(simu_dir):
    _, samples, chrom, chrom_len = _init_vcf(str(simu_dir / "chr21.vcf.gz"), 1)
    assert samples == ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
    assert chrom == "chr21"          # header lists chr20 first
    assert chrom_len == 46709983


def test_read_simu_counts_semantics(simu_dir):
    loci_df, g_anc, local_array = read_simu(str(simu_dir), verbose=False)

    assert loci_df["physical_position"].tolist() == [100, 5000, 1500000]
    assert loci_df["i"].tolist() == [0, 1, 2]
    assert local_array.dtype == np.int8
    assert local_array.shape == (3, 4, 3)     # (variants, samples, ancestries)
    np.testing.assert_array_equal(local_array.compute(), EXPECTED)

    assert list(get_pops(g_anc)) == ["CEU", "NAT", "YRI"]
    assert g_anc["sample_id"].tolist() == ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
    np.testing.assert_allclose(g_anc[["CEU", "NAT", "YRI"]].sum(axis=1), 1.0, atol=1e-6)
    # Sample_1: YRI,YRI / YRI,CEU / CEU,CEU -> CEU 3/6, YRI 3/6
    np.testing.assert_allclose(g_anc.loc[0, ["CEU", "NAT", "YRI"]].to_numpy(dtype=float), [0.5, 0, 0.5])


def test_read_simu_region_order_is_genomic(simu_dir):
    """Regions are pulled in parallel; rows must still follow loci order."""
    loci_df, _, local_array = read_simu(str(simu_dir), verbose=False,
                                        chunk_size=1_000_000, n_threads=4)
    # the third record lives in the second 1 Mb region
    np.testing.assert_array_equal(local_array.compute()[2], EXPECTED[2])


def test_read_simu_chrom_filter_and_missing(simu_dir):
    _, _, arr = read_simu(str(simu_dir), verbose=False, chrom="chr21")
    assert arr.shape[0] == 3
    with pytest.raises(FileNotFoundError):
        read_simu(str(simu_dir), verbose=False, chrom="22")


def test_read_simu_stress(tmp_path):
    """~50 samples, bgzip + tabix built on the fly."""
    pysam = pytest.importorskip("pysam")
    n_samples = 50
    samples = [f"Sample_{i}" for i in range(1, n_samples + 1)]
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##contig=<ID=chr21,length=46709983>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=POP,Number=2,Type=String,Description="Origin Population">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}
    """.format("\t".join(samples)))
    pops = ["0|0:YRI,YRI" if i % 2 == 0 else "0|0:CEU,YRI" for i in range(n_samples)]
    record = "chr21\t5030578\t21:5030578:C:T\tC\tT\t.\tPASS\t.\tGT:POP\t" + "\t".join(pops)
    vcf_path = tmp_path / "chr21.vcf"
    vcf_path.write_text(header + record + "\n")
    gz = str(vcf_path) + ".gz"
    pysam.tabix_compress(str(vcf_path), gz, force=True)
    pysam.tabix_index(gz, preset="vcf", force=True)
    os.remove(vcf_path)
    (tmp_path / "chr21.bp").write_text("Sample_1_1\nYRI\t21\t1\t0\nCEU\t21\t2\t0\n")

    loci_df, g_anc, local_array = read_simu(str(tmp_path), verbose=False)
    assert loci_df.shape[0] == 1
    assert local_array.shape == (1, n_samples, 2)
    counts = local_array.compute()[0]
    np.testing.assert_array_equal(counts[0], [0, 2])   # YRI,YRI -> CEU=0, YRI=2
    np.testing.assert_array_equal(counts[1], [1, 1])
    assert np.allclose(g_anc[["CEU", "YRI"]].sum(axis=1).values, 1.0, atol=1e-6)
