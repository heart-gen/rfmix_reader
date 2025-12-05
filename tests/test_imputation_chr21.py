from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from rfmix_reader import interpolate_array, read_rfmix
from rfmix_reader.processing.imputation import GPU_ENABLED
from rfmix_reader.readers.read_rfmix import gpu_available


@pytest.mark.filterwarnings("ignore:.*cupy not installed.*")
@pytest.mark.parametrize("method", ["linear", "nearest", "stepwise"])
def test_imputation_chr21_interpolation(tmp_path, method):
    loci_df, g_anc, admix = read_rfmix(
        "data/",
        binary_dir=tmp_path / "binary",
        generate_binary=True,
        verbose=False,
    )

    loci_pd = loci_df.to_pandas() if hasattr(loci_df, "to_pandas") else loci_df.copy()
    renamed = loci_pd.rename(columns={"chromosome": "chrom", "physical_position": "pos"})

    chrom = renamed["chrom"].iloc[0]
    first_two = renamed["pos"].iloc[:2].to_numpy()
    midpoint = int(np.mean(first_two))
    while midpoint in set(renamed["pos"].to_numpy()):
        midpoint += 1

    missing_row = pd.DataFrame({
        "chrom": [chrom],
        "pos": [midpoint],
        "i": [np.nan],
    })

    variant_loci_df = (
        pd.concat([renamed.loc[:, ["chrom", "pos", "i"]], missing_row], ignore_index=True)
        .sort_values("pos")
        .reset_index(drop=True)
    )

    z = interpolate_array(
        variant_loci_df,
        admix,
        zarr_outdir=tmp_path / f"zarr-{method}",
        chunk_size=500,
        batch_size=2000,
        interpolation=method,
        use_bp_positions=True,
    )

    assert z.shape == (len(variant_loci_df), admix.shape[1], admix.shape[2])
    assert not np.isnan(z[:]).any()

    if gpu_available():
        if not GPU_ENABLED:
            pytest.skip("CUDA is available but CuPy is missing; GPU interpolation cannot be tested.")
    else:
        assert not GPU_ENABLED
