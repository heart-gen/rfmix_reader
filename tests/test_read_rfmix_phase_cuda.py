from pathlib import Path

import numpy as np
import pytest

from rfmix_reader.io.prepare_reference import convert_vcf_to_zarr
from rfmix_reader.processing.phase import phase_rfmix_chromosome_to_zarr
from rfmix_reader.formats.common import read_rfmix_q


def gpu_available() -> bool:
    try:
        from torch.cuda import is_available
    except ImportError:
        return False
    return bool(is_available())


@pytest.mark.cuda
@pytest.mark.skipif(not gpu_available(), reason="CUDA is required for this integration test")
def test_reference_phasing_with_vcf_zarr(tmp_path, request, chr21_lfs):
    """End-to-end ``method="reference"`` phasing against a 1kGP VCF-Zarr panel."""
    if not request.config.getoption("--run-cuda-tests"):
        pytest.skip("CUDA integration tests run only with --run-cuda-tests")

    ref_vcf = chr21_lfs / "1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
    if not ref_vcf.exists():
        pytest.skip("Reference VCF for Zarr conversion is not available")

    ref_zarr = tmp_path / "ref_panel.zarr"
    convert_vcf_to_zarr(str(ref_vcf), str(ref_zarr), verbose=False)

    q_df = read_rfmix_q(str(chr21_lfs / "chr21.rfmix.Q"), add_chrom=False)
    pop_cols = [c for c in q_df.columns if c != "sample_id"]
    sample_annot = q_df[["sample_id"]].copy()
    sample_annot["group"] = q_df[pop_cols].idxmax(axis=1)
    annot_path = tmp_path / "annot.tsv"
    sample_annot.to_csv(annot_path, sep="\t", index=False)

    out = phase_rfmix_chromosome_to_zarr(
        str(chr21_lfs), str(ref_zarr), str(annot_path), str(tmp_path / "phased.zarr"),
        chrom="21", method="reference", verbose=False,
    )
    assert out.sizes["sample"] == len(q_df)
    assert Path(tmp_path / "phased.zarr").is_dir()
    assert np.isin(out.haplotype_ancestry.values, [-1, 0, 1]).all()
