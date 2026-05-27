from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
da = pytest.importorskip("dask.array")

from rfmix_reader import interpolate_array, read_rfmix_fb
from rfmix_reader.processing.imputation import (
    GPU_ENABLED, _expand_array, _interpolate_1d, interpolate_block,
)
from rfmix_reader.readers.read_rfmix import gpu_available


@pytest.mark.filterwarnings("ignore:.*cupy not installed.*")
@pytest.mark.parametrize("method", ["linear", "nearest", "stepwise"])
def test_imputation_chr21_interpolation(tmp_path, method):
    loci_df, g_anc, admix = read_rfmix_fb(
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


def test_gpu_enabled_importable():
    """GPU_ENABLED must be importable (Bug 2 regression)."""
    assert isinstance(GPU_ENABLED, bool)


@pytest.mark.parametrize("method", ["stepwise", "nearest", "linear"])
def test_stepwise_boundary(method):
    """
    Leading and trailing NaNs must be filled for each interpolation method.

    Column: NaN NaN 1 NaN 1 NaN NaN  (shape 7×1×1)
    - All methods: fill leading NaNs with first valid value (1.0)
    - All methods: fill trailing NaNs with last valid value (1.0)
    - Interior NaN (index 3): filled by each method (all valid between 1 and 1 → 1)
    """
    col = np.array([np.nan, np.nan, 1.0, np.nan, 1.0, np.nan, np.nan],
                   dtype=np.float32)
    block = col.reshape(7, 1, 1)
    out = interpolate_block(block, method=method)
    assert not np.isnan(out).any(), f"NaNs remain after {method} interpolation"
    result = out[:, 0, 0]
    # All valid values in this column are 1.0, so every filled position must be 1
    np.testing.assert_array_equal(
        result, np.ones(7, dtype=np.float32),
        err_msg=f"[{method}] expected all-ones after filling NaNs, got {result}",
    )


def test_expand_array_unsorted_raises(tmp_path):
    """
    _expand_array must raise ValueError when 'i' values are not sorted (Bug 1).
    """
    import dask.array as da

    # admix: 3 loci × 2 samples × 1 ancestry, all zeros
    admix = da.zeros((3, 2, 1), dtype=np.float32, chunks=(3, 2, 1))

    # variant_loci_df with i values out of order: 2, 0, 1
    variant_loci_df = pd.DataFrame({
        "pos": [100, 200, 300],
        "i": [2.0, 0.0, 1.0],
    })

    with pytest.raises(ValueError, match="monotonically"):
        _expand_array(variant_loci_df, admix, zarr_outdir=str(tmp_path / "zarr"))


def test_expand_array_slab_path_correctness(tmp_path, monkeypatch):
    """
    Slab path (used when admix doesn't fit in memory) must write correct values (Bug 1 fix).

    Force slab path by monkeypatching psutil to report 0 available memory.
    """
    import psutil
    import dask.array as da

    # admix: 4 loci × 2 samples × 1 ancestry; values 0,1,2,3
    np_data = np.arange(4 * 2 * 1, dtype=np.float32).reshape(4, 2, 1)
    admix = da.from_array(np_data, chunks=(4, 2, 1))

    # variant_loci_df: 5 rows — 4 valid (sorted) + 1 NaN (missing locus)
    variant_loci_df = pd.DataFrame({
        "pos": [100, 200, 300, 400, 500],
        "i": [0.0, 1.0, 2.0, np.nan, 3.0],
    })

    class FakeVMResult:
        available = 0  # force slab path

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVMResult())

    zarr_dir = tmp_path / "zarr_slab"
    z = _expand_array(variant_loci_df, admix, zarr_outdir=str(zarr_dir))

    # Row 3 (i=NaN) should be NaN; all others should match admix
    assert np.isnan(z[3, :, :]).all(), "Missing locus row should be NaN"
    for dest, src in [(0, 0), (1, 1), (2, 2), (4, 3)]:
        np.testing.assert_array_equal(
            z[dest, :, :], np_data[src, :, :],
            err_msg=f"Slab path: dest row {dest} (src {src}) mismatch",
        )


def test_imputation_ignores_nan_metadata(tmp_path):
    loci_df, _, admix = read_rfmix(
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

    missing_row = pd.DataFrame({"chrom": [chrom], "pos": [midpoint], "i": [np.nan]})

    variant_loci_df = (
        pd.concat([renamed.loc[:, ["chrom", "pos", "i"]], missing_row], ignore_index=True)
        .assign(annotation=np.nan)
        .sort_values("pos")
        .reset_index(drop=True)
    )

    z = interpolate_array(
        variant_loci_df,
        admix,
        zarr_outdir=tmp_path / "zarr-ignore-nan",
        chunk_size=500,
        batch_size=2000,
        interpolation="linear",
        use_bp_positions=True,
    )

    assert z.shape == (len(variant_loci_df), admix.shape[1], admix.shape[2])
    assert not np.isnan(z[:]).any()
