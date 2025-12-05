from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
xr = pytest.importorskip("xarray")

from rfmix_reader.io.prepare_reference import convert_vcf_to_zarr
from rfmix_reader.processing.phase import phase_rfmix_chromosome_to_zarr
from rfmix_reader.readers.read_rfmix import _read_Q_noi, gpu_available


@pytest.mark.skipif(
    not gpu_available(),
    reason="CUDA is required for phased RFMix integration test",
)
def test_read_rfmix_phase_cuda(tmp_path, request):
    if not request.config.getoption("--run-cuda-tests"):
        pytest.skip("CUDA integration tests run only with --run-cuda-tests")

    ref_vcf = Path("data/1kGP_high_coverage_Illumina.chr21.filtered.SNV_INDEL_SV_phased_panel.vcf.gz")

    if not ref_vcf.exists():
        pytest.skip("Reference VCF for Zarr conversion is not available")

    ref_zarr = tmp_path / "ref_panel.zarr"
    convert_vcf_to_zarr(str(ref_vcf), str(ref_zarr), verbose=False)

    q_path = "data/chr21.rfmix.Q"
    q_df = _read_Q_noi(q_path)

    pop_cols = [col for col in q_df.columns if col != "sample_id"]
    sample_annot = q_df[["sample_id"]].copy()
    sample_annot["group"] = q_df[pop_cols].idxmax(axis=1)

    sample_annot_path = tmp_path / "sample_annot.tsv"
    sample_annot.to_csv(str(sample_annot_path), sep="\t", index=False, header=False)

    binary_dir = tmp_path / "binary"
    out_path = tmp_path / "phased_chr21.zarr"
    ds = phase_rfmix_chromosome_to_zarr(
        file_prefix="data/chr21",
        binary_dir=str(binary_dir),
        generate_binary=True,
        ref_zarr_root=str(ref_zarr),
        sample_annot_path=str(sample_annot_path),
        output_path=str(out_path),
        chrom="21",
    )

    nloci = ds.dims["variant"]
    nsamples = q_df.shape[0]
    npops = len(pop_cols)

    assert nloci == ds.local_ancestry.shape[0]
    assert nsamples == ds.local_ancestry.shape[1]
    assert npops == ds.local_ancestry.shape[2]

    reopened = xr.open_zarr(out_path)
    sample_slice = reopened.local_ancestry.isel(sample=0).isel(
        variant=slice(0, min(10, nloci))
    ).data.compute()
    assert np.isfinite(sample_slice).all()
    assert set(np.unique(sample_slice)).issubset({0, 1, 2})
