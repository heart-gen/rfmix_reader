import textwrap
import numpy as np
import pytest
import pandas as pd
import os

from rfmix_reader._read_simu import (
    read_simu,
    _parse_pop_labels,
    _map_pop_to_codes,
    _build_mapper,
    MISSING,
)

@pytest.fixture
def realistic_vcf(tmp_path):
    """Create a VCF with GT:POP format field and multiple samples."""
    vcf_path = tmp_path / "chr21.vcf"
    bp_path = tmp_path / "chr21.bp"

    # Build header with realistic fields
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##contig=<ID=chr21>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=POP,Number=2,Type=String,Description="Origin Population of each respective allele in GT">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1\tSample_2\tSample_3\tSample_4\tSample_5
    """)

    # One record with GT:POP values
    record = (
        "chr21\t5030578\t21:5030578:C:T\tC\tT\t.\tPASS\t.\tGT:POP\t"
        "0|0:YRI,YRI\t0|1:YRI,YRI\t0|0:YRI,YRI\t0|0:YRI,CEU\t1|0:CEU,YRI"
    )

    with open(vcf_path, "w") as f:
        f.write(header)
        f.write(record + "\n")

    # .bp file with ancestry labels
    with open(bp_path, "w") as f:
        f.write("YRI\nCEU\n")

    return str(vcf_path)


def test_parse_pop_labels_bp(realistic_vcf):
    labels = _parse_pop_labels(realistic_vcf)
    assert labels == ["CEU", "YRI"]


def test_parse_pop_labels_fallback(tmp_path):
    """Remove .bp and ensure fallback to VCF POP works."""
    vcf_path = tmp_path / "chr21.vcf"
    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##contig=<ID=chr21>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=POP,Number=2,Type=String,Description="Origin Population of each respective allele in GT">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1
    """)
    record = "chr21\t5030578\t21:5030578:C:T\tC\tT\t.\tPASS\t.\tGT:POP\t0|0:YRI,CEU"

    with open(vcf_path, "w") as f:
        f.write(header)
        f.write(record + "\n")

    labels = _parse_pop_labels(str(vcf_path))
    assert "YRI" in labels and "CEU" in labels


def test_map_pop_to_codes():
    ancestries, mapper = _build_mapper(["CEU", "YRI"])
    arr = np.array([["YRI,CEU", "CEU,YRI"]])
    codes = _map_pop_to_codes(arr, ancestries)
    assert codes.shape == (1, 2, 2)
    assert np.all(codes != MISSING)


def test_read_simu_runs(realistic_vcf):
    loci_df, g_anc, local_array = read_simu(os.path.dirname(realistic_vcf))
    assert loci_df.shape[0] == 1
    assert g_anc.shape[0] == 5
    assert set(["CEU", "YRI"]).issubset(set(g_anc.columns))
    assert local_array.shape == (1, 5, 2)


def test_read_simu_stress(tmp_path):
    """Stress test with ~50 samples and multiple POP labels."""
    vcf_path = tmp_path / "chr21.vcf"
    bp_path = tmp_path / "chr21.bp"

    n_samples = 50
    samples = [f"Sample_{i}" for i in range(1, n_samples + 1)]

    header = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##contig=<ID=chr21>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
        ##FORMAT=<ID=POP,Number=2,Type=String,Description="Origin Population of each respective allele in GT">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}
    """.format("\t".join(samples)))

    pops = []
    for i in range(n_samples):
        if i % 2 == 0:
            pops.append("0|0:YRI,YRI")
        else:
            pops.append("0|0:CEU,YRI")
    record = "chr21\t5030578\t21:5030578:C:T\tC\tT\t.\tPASS\t.\tGT:POP\t{}".format("\t".join(pops))

    with open(vcf_path, "w") as f:
        f.write(header)
        f.write(record + "\n")

    with open(bp_path, "w") as f:
        f.write("YRI\nCEU\n")

    loci_df, g_anc, local_array = read_simu(os.path.dirname(vcf_path))

    assert loci_df.shape[0] == 1
    assert g_anc.shape[0] == n_samples
    assert local_array.shape == (1, n_samples, 2)
    assert np.allclose(g_anc[["YRI", "CEU"]].sum(axis=1).values, 1.0, atol=1e-6)
