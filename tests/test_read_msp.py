"""
Unit tests for the read_rfmix reader (reads .msp.tsv files).
"""
import gzip
import pytest
import importlib

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
da = pytest.importorskip("dask.array")

msp_mod = importlib.import_module("rfmix_reader.readers.read_msp")
_parse_pop_header = msp_mod._parse_pop_header
_read_msp_file = msp_mod._read_msp_file
_segments_to_loci = msp_mod._segments_to_loci
_get_msp_prefixes = msp_mod._get_msp_prefixes
extract_locus_ancestry = msp_mod.extract_locus_ancestry
read_rfmix = msp_mod.read_rfmix

_MSP_CONTENT = """\
#Subpopulation order/codes: AFR=0\tEUR=1
#chm\tspos\tepos\tsgpos\tegpos\tn snps\tSample_1.0\tSample_1.1\tSample_2.0\tSample_2.1
chr1\t10000\t50000\t0.00\t0.10\t100\t0\t1\t1\t1
chr1\t50001\t90000\t0.10\t0.20\t80\t1\t0\t0\t1
"""


@pytest.fixture
def msp_file(tmp_path):
    fn = tmp_path / "chr1.msp.tsv"
    fn.write_text(_MSP_CONTENT)
    return str(fn)


# ---------------------------------------------------------------------------
# _parse_pop_header
# ---------------------------------------------------------------------------

def test_parse_pop_header(msp_file):
    pop_map = _parse_pop_header(msp_file)
    assert pop_map == {"AFR": 0, "EUR": 1}


def test_parse_pop_header_invalid(tmp_path):
    fn = tmp_path / "bad.msp.tsv"
    fn.write_text("no codes here\n#chm spos\n")
    with pytest.raises(ValueError, match="Could not parse"):
        _parse_pop_header(str(fn))


# ---------------------------------------------------------------------------
# _read_msp_file
# ---------------------------------------------------------------------------

def test_read_msp_file_shape(msp_file):
    segs, hap_cols, pop_map = _read_msp_file(msp_file)
    assert segs.shape[0] == 2  # 2 segments
    assert len(hap_cols) == 4  # 2 samples × 2 haplotypes
    assert pop_map == {"AFR": 0, "EUR": 1}


def test_read_msp_file_columns(msp_file):
    segs, hap_cols, _ = _read_msp_file(msp_file)
    assert "chrom" in segs.columns
    assert "spos" in segs.columns
    assert "epos" in segs.columns
    assert set(hap_cols) == {"Sample_1.0", "Sample_1.1", "Sample_2.0", "Sample_2.1"}


# ---------------------------------------------------------------------------
# _segments_to_loci
# ---------------------------------------------------------------------------

def test_segments_to_loci_shape(msp_file):
    segs, hap_cols, pop_map = _read_msp_file(msp_file)
    segs_pd = segs.to_pandas() if hasattr(segs, "to_pandas") else segs
    loci_df, local_array = _segments_to_loci(segs_pd, hap_cols, pop_map)

    assert loci_df.shape == (2, 3)  # (n_segs, [chromosome, physical_position, i])
    assert local_array.shape == (2, 2, 2)  # (n_segs, n_samples, n_pops)


def test_segments_to_loci_ancestry_counts(msp_file):
    """
    Verify diploid ancestry counts from haplotype pairs.

    Segment 0: Sample_1 (AN=0,1) → AFR=1,EUR=1;  Sample_2 (AN=1,1) → AFR=0,EUR=2
    Segment 1: Sample_1 (AN=1,0) → AFR=1,EUR=1;  Sample_2 (AN=0,1) → AFR=1,EUR=1

    Populations sorted alphabetically: AFR=axis0, EUR=axis1.
    """
    segs, hap_cols, pop_map = _read_msp_file(msp_file)
    segs_pd = segs.to_pandas() if hasattr(segs, "to_pandas") else segs
    loci_df, local_array = _segments_to_loci(segs_pd, hap_cols, pop_map)
    result = local_array.compute()

    # Segment 0
    np.testing.assert_array_equal(result[0, 0, :], [1, 1],
                                  err_msg="Seg0 Sample_1: expected AFR=1,EUR=1")
    np.testing.assert_array_equal(result[0, 1, :], [0, 2],
                                  err_msg="Seg0 Sample_2: expected AFR=0,EUR=2")

    # Segment 1
    np.testing.assert_array_equal(result[1, 0, :], [1, 1],
                                  err_msg="Seg1 Sample_1: expected AFR=1,EUR=1")
    np.testing.assert_array_equal(result[1, 1, :], [1, 1],
                                  err_msg="Seg1 Sample_2: expected AFR=1,EUR=1")


def test_segments_to_loci_index_offset(msp_file):
    segs, hap_cols, pop_map = _read_msp_file(msp_file)
    segs_pd = segs.to_pandas() if hasattr(segs, "to_pandas") else segs
    loci_df, _ = _segments_to_loci(segs_pd, hap_cols, pop_map, index_offset=10)
    assert list(loci_df["i"]) == [10, 11]


# ---------------------------------------------------------------------------
# read_rfmix end-to-end (reads .msp.tsv)
# ---------------------------------------------------------------------------

def test_read_rfmix_returns_triple(tmp_path):
    fn = tmp_path / "chr1.msp.tsv"
    fn.write_text(_MSP_CONTENT)

    loci_df, g_anc_out, local_array = read_rfmix(str(tmp_path), verbose=False)

    assert isinstance(local_array, da.Array)
    assert loci_df.shape[0] == 2
    assert local_array.shape == (2, 2, 2)
    assert g_anc_out is None  # no g_anc supplied


def test_read_rfmix_passthrough_g_anc(tmp_path):
    fn = tmp_path / "chr1.msp.tsv"
    fn.write_text(_MSP_CONTENT)

    dummy_g_anc = pd.DataFrame({"sample_id": ["S1"], "AFR": [0.5], "EUR": [0.5]})
    _, g_anc_out, _ = read_rfmix(str(tmp_path), g_anc=dummy_g_anc, verbose=False)

    assert g_anc_out is dummy_g_anc


def test_get_msp_prefixes_accepts_path_prefix(tmp_path):
    fn = tmp_path / "run_chr1.msp.tsv"
    fn.write_text(_MSP_CONTENT)

    result = _get_msp_prefixes(str(tmp_path / "run_chr1"), verbose=False)

    assert result == [{"msp.tsv": str(fn)}]


_Q_CONTENT = """\
#rfmix diploid global ancestry .Q format output
#sample	AFR	EUR
Sample_1	0.5	0.5
Sample_2	0.25	0.75
"""


def test_get_msp_prefixes_accepts_gzip(tmp_path):
    fn = tmp_path / "chr1.msp.tsv.gz"
    with gzip.open(fn, "wt") as fh:
        fh.write(_MSP_CONTENT)

    result = _get_msp_prefixes(str(tmp_path), verbose=False)

    assert result == [{"msp.tsv": str(fn)}]


def test_read_rfmix_reads_neighboring_q(tmp_path):
    (tmp_path / "chr1.msp.tsv").write_text(_MSP_CONTENT)
    (tmp_path / "chr1.rfmix.Q").write_text(_Q_CONTENT)

    _, g_anc_out, _ = read_rfmix(str(tmp_path), verbose=False)

    assert g_anc_out is not None
    assert list(g_anc_out.columns) == ["sample_id", "AFR", "EUR", "chrom"]
    assert list(g_anc_out["sample_id"]) == ["Sample_1", "Sample_2"]
    assert list(g_anc_out["chrom"]) == ["chr1", "chr1"]


def test_read_rfmix_can_skip_q(tmp_path):
    (tmp_path / "chr1.msp.tsv").write_text(_MSP_CONTENT)
    (tmp_path / "chr1.rfmix.Q").write_text(_Q_CONTENT)

    _, g_anc_out, _ = read_rfmix(str(tmp_path), verbose=False, read_q=False)

    assert g_anc_out is None


def test_extract_locus_ancestry_aggregate_boundary_and_outside(tmp_path):
    (tmp_path / "chr1.msp.tsv").write_text(_MSP_CONTENT)
    loci = pd.DataFrame({
        "variant_id": ["inside", "boundary", "outside"],
        "chrom": ["1", "chr1", "chr1"],
        "pos": [10001, 50000, 90001],
    })

    out = extract_locus_ancestry(str(tmp_path), loci, aggregate=True)

    assert list(out["variant_id"]) == ["inside", "boundary", "outside"]
    assert list(out["matched"]) == [True, True, False]
    assert out.loc[0, "AFR_haplotypes"] == 1
    assert out.loc[0, "EUR_haplotypes"] == 3
    assert out.loc[1, "AFR_haplotypes"] == 1
    assert out.loc[1, "EUR_fraction"] == 0.75
    assert pd.isna(out.loc[2, "AFR_fraction"])


def test_extract_locus_ancestry_sample_level_subset(tmp_path):
    (tmp_path / "chr1.msp.tsv").write_text(_MSP_CONTENT)
    loci = pd.DataFrame({"chrom": ["chr1"], "pos": [50001], "rsid": ["rs1"]})

    out = extract_locus_ancestry(
        str(tmp_path), loci, samples=["Sample_2"], aggregate=False
    )

    assert out.shape[0] == 1
    assert out.loc[0, "sample_id"] == "Sample_2"
    assert out.loc[0, "AFR_copies"] == 1
    assert out.loc[0, "EUR_copies"] == 1
