"""
Unit tests for rfmix_reader.io.write_data internals.
Covers: Bug 4 (_debug_partition_alignment raises RuntimeError, not SystemExit),
        Memory 3 (Zarr chunk alignment in _clean_data_imp).
"""
import pytest
import importlib

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
da = pytest.importorskip("dask.array")
zarr = pytest.importorskip("zarr")

wd = importlib.import_module("rfmix_reader.io.write_data")
_debug_partition_alignment = wd._debug_partition_alignment
_clean_data_imp = wd._clean_data_imp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_admix_arr(nrows, n_samples=2, n_anc=2, chunk=10):
    data = np.zeros((nrows, n_samples, n_anc), dtype=np.float32)
    return da.from_array(data, chunks=(min(chunk, nrows), n_samples, n_anc))


def _make_loci_ddf(nrows, npartitions=2):
    import dask.dataframe as dd
    df = pd.DataFrame({
        "chrom": ["chr1"] * nrows,
        "pos": list(range(nrows)),
    })
    return dd.from_pandas(df, npartitions=npartitions)


# ---------------------------------------------------------------------------
# Bug 4: _debug_partition_alignment raises RuntimeError, not SystemExit
# ---------------------------------------------------------------------------

class TestDebugPartitionAlignment:
    def test_raises_runtime_error_on_block_mismatch(self):
        """Must raise RuntimeError (not SystemExit) when block counts differ."""
        admix_arr = _make_admix_arr(20, chunk=10)   # 2 blocks
        loci_ddf = _make_loci_ddf(20, npartitions=3)  # 3 partitions

        with pytest.raises(RuntimeError, match="Number of partitions"):
            _debug_partition_alignment(admix_arr, loci_ddf, verbose=False)

    def test_raises_runtime_error_on_row_mismatch(self):
        """Must raise RuntimeError when row counts per partition differ."""
        # admix: 2 blocks of 10 rows each
        admix_arr = _make_admix_arr(20, chunk=10)
        # loci_ddf: 2 partitions but with 12 and 8 rows
        import dask.dataframe as dd
        df = pd.DataFrame({
            "chrom": ["chr1"] * 20,
            "pos": list(range(20)),
        })
        # Force unequal partitions by constructing manually
        ddf1 = dd.from_pandas(df.iloc[:12], npartitions=1)
        ddf2 = dd.from_pandas(df.iloc[12:], npartitions=1)
        loci_ddf = dd.concat([ddf1, ddf2])

        with pytest.raises(RuntimeError, match="Mismatch in partition"):
            _debug_partition_alignment(admix_arr, loci_ddf, verbose=True)

    def test_passes_when_aligned(self):
        """Must not raise when block count and row counts match."""
        admix_arr = _make_admix_arr(20, chunk=10)  # 2 blocks of 10
        loci_ddf = _make_loci_ddf(20, npartitions=2)

        # Should not raise
        _debug_partition_alignment(admix_arr, loci_ddf, verbose=False)


# ---------------------------------------------------------------------------
# Memory 3: from_array on Zarr uses z.chunks (native), not admix.chunksize
# ---------------------------------------------------------------------------

class TestCleanDataImpChunkAlignment:
    def test_zarr_native_chunks_via_from_array(self, tmp_path):
        """
        When wrapping a Zarr with dask.array.from_array, using z.chunks (native
        Zarr chunk shape) avoids cross-chunk read amplification (Memory 3 fix).

        This test verifies directly that from_array(z, chunks=z.chunks) produces
        chunks aligned with the Zarr file, and that the result is distinct from
        the misaligned case where admix.chunksize is used instead.
        """
        from dask.array import from_array

        zarr_chunks = (4, 2, 2)
        z = zarr.open(
            str(tmp_path / "test.zarr"), mode="w",
            shape=(8, 2, 2), chunks=zarr_chunks, dtype="float32",
            fill_value=np.nan,
        )
        z[:] = np.ones((8, 2, 2), dtype=np.float32)

        # Correct (fixed) path: use z.chunks
        daz_correct = from_array(z, chunks=z.chunks)
        assert daz_correct.chunks[0] == (4, 4), (
            f"Expected loci chunks (4, 4) from Zarr native, got {daz_correct.chunks[0]}"
        )

        # Misaligned (old) path: admix chunk shape does not match Zarr
        misaligned_chunk = (3, 2, 2)  # does not divide 8 evenly into zarr_chunks
        daz_wrong = from_array(z, chunks=misaligned_chunk)
        # daz_wrong has chunks (3, 3, 2) along axis 0 — misaligned with zarr_chunks[0]=4
        assert daz_wrong.chunks[0] != daz_correct.chunks[0], (
            "Wrong and correct paths should produce different chunk structures"
        )

    def test_clean_data_imp_runs_without_error(self, tmp_path):
        """_clean_data_imp must complete successfully with the fixed chunk path."""
        zarr_chunks = (4, 2, 2)
        z = zarr.open(
            str(tmp_path / "test.zarr"), mode="w",
            shape=(8, 2, 2), chunks=zarr_chunks, dtype="float32",
            fill_value=np.nan,
        )
        z[:] = np.ones((8, 2, 2), dtype=np.float32)

        admix = _make_admix_arr(8, n_samples=2, n_anc=2, chunk=3)
        variant_loci = pd.DataFrame({
            "chrom": ["chr1"] * 8,
            "pos": list(range(8)),
            "i": list(range(8)),
            "_merge": ["both"] * 8,
        })

        loci_I, admix_I = _clean_data_imp(admix, variant_loci, z)
        assert loci_I.shape[0] == 8
        result = admix_I.compute()
        assert result.shape == (8, 2, 2)
        np.testing.assert_array_equal(result, np.ones((8, 2, 2), dtype=np.float32))
