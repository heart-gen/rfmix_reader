"""Schema, accessor and Zarr store round trips on tiny synthetic data."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rfmix_reader.core import build_dataset, concat_datasets, open_store, validate, write_store
from rfmix_reader.formats.base import Chunk, Header


def _codes():
    # 5 variants, 2 samples: S1 = EUR/EUR then EUR/AFR ..., S2 has a missing hap at row 2
    codes = np.zeros((5, 2, 2), dtype=np.int8)
    codes[:, 0, 1] = [0, 1, 1, 1, 0]
    codes[:, 1, :] = [[1, 1], [1, 0], [-1, 0], [0, 0], [1, 1]]
    return codes


def _dataset(**kw):
    codes = _codes()
    return build_dataset(
        ["chr1"] * 5, np.arange(100, 600, 100), None, codes, ["S1", "S2"], ["EUR", "AFR"],
        global_ancestry=np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32),
        source_format="test", **kw,
    )


def test_build_dataset_schema_and_validate():
    ds = _dataset()
    validate(ds)
    assert dict(ds.sizes) == {"variant": 5, "sample": 2, "ploidy": 2, "ancestry": 2, "contig": 1}
    assert ds.haplotype_ancestry.dtype == np.int8
    assert ds.la.samples == ["S1", "S2"] and ds.la.ancestries == ["EUR", "AFR"]
    assert ds.la.chromosomes == ["chr1"]
    assert ds.contig.values.tolist() == ["chr1"]
    assert ds.attrs["source_format"] == "test"
    np.testing.assert_array_equal(ds.segment_end.values, ds.variant_position.values)


def test_build_dataset_shape_errors():
    with pytest.raises(ValueError, match="haplotype_ancestry"):
        build_dataset(["chr1"] * 5, np.arange(5), None, np.zeros((5, 3, 2), np.int8), ["S1", "S2"], ["A", "B"])
    with pytest.raises(ValueError, match="chromosome"):
        build_dataset(["chr1"] * 4, np.arange(5), None, np.zeros((5, 2, 2), np.int8), ["S1", "S2"], ["A", "B"])


def test_validate_rejects_bad_dims():
    ds = _dataset()
    bad = ds.assign(haplotype_ancestry=ds.haplotype_ancestry.astype(np.int32))
    with pytest.raises(ValueError, match="int8"):
        validate(bad)
    with pytest.raises(ValueError, match="no 'haplotype_ancestry'"):
        validate(xr.Dataset())


def test_accessor_counts_and_missing():
    ds = _dataset()
    counts = ds.la.counts
    assert tuple(counts.dims) == ("variant", "sample", "ancestry")
    c = counts.values
    assert c.dtype == np.int8
    np.testing.assert_array_equal(c[:, 0, :], [[2, 0], [1, 1], [1, 1], [1, 1], [2, 0]])
    np.testing.assert_array_equal(c[2, 1, :], [-1, -1])
    np.testing.assert_array_equal(c[0, 1, :], [0, 2])


def test_accessor_global_ancestry_and_legacy():
    ds = _dataset()
    g = ds.la.global_ancestry
    assert list(g.columns) == ["sample_id", "EUR", "AFR", "chrom"]
    assert g["chrom"].tolist() == ["chr1", "chr1"]
    loci, g2, arr = ds.la.to_legacy()
    assert list(loci.columns) == ["chromosome", "physical_position", "i"]
    assert loci["i"].tolist() == [0, 1, 2, 3, 4]
    assert loci["physical_position"].dtype == np.int32
    np.testing.assert_array_equal(arr.compute(), ds.la.counts.values)
    pd.testing.assert_frame_equal(g2, g)


def test_accessor_sel_region():
    ds = _dataset()
    sub = ds.la.sel_region("1", 200, 400)
    assert sub.variant_position.values.tolist() == [200, 300, 400]
    assert ds.la.sel_region("chr2").sizes["variant"] == 0


def test_concat_datasets_two_chromosomes():
    a = _dataset()
    b = build_dataset(["chr2"] * 5, np.arange(10, 60, 10), None, _codes(), ["S1", "S2"], ["EUR", "AFR"],
                      global_ancestry=np.full((2, 2), 0.5, np.float32), source_format="test")
    ds = concat_datasets([a, b])
    assert ds.sizes["variant"] == 10 and ds.sizes["contig"] == 2
    assert ds.la.chromosomes == ["chr1", "chr2"]
    assert ds.la.global_ancestry["chrom"].tolist() == ["chr1"] * 2 + ["chr2"] * 2
    with pytest.raises(ValueError, match="Sample"):
        concat_datasets([a, b.assign_coords(sample_id=("sample", ["X", "Y"]))])
    with pytest.raises(ValueError, match="Ancestry"):
        concat_datasets([a, b.assign_coords(ancestry=("ancestry", ["AFR", "EUR"]))])


def test_write_store_round_trip(tmp_path):
    codes = _codes()
    header = Header(samples=["S1", "S2"], ancestries=["EUR", "AFR"], source_files=["x.msp.tsv"],
                    chrom="chr1", global_ancestry=None)
    chunks = [
        Chunk(np.array(["chr1"] * 3), np.array([100, 200, 300], np.int32),
              np.array([150, 250, 350], np.int32), codes[:3]),
        Chunk(np.array(["chr1"] * 2), np.array([400, 500], np.int32),
              np.array([450, 550], np.int32), codes[3:]),
    ]
    path = write_store(header, chunks, tmp_path / "chr1.zarr", chunk_rows=2, source_format="test")
    ds = open_store(path)
    validate(ds)
    assert ds.sizes["variant"] == 5
    assert ds.haplotype_ancestry.data.chunks[0] == (2, 2, 1)
    np.testing.assert_array_equal(ds.haplotype_ancestry.values, codes)
    assert ds.chromosome.values.tolist() == ["chr1"] * 5
    assert ds.segment_end.values.tolist() == [150, 250, 350, 450, 550]
    assert ds.la.samples == ["S1", "S2"] and ds.la.ancestries == ["EUR", "AFR"]
    assert ds.attrs["source_format"] == "test" and ds.attrs["n_variants"] == 5
    assert ds.la.posterior is None
    # global ancestry computed from the codes: S1 has 7 EUR / 3 AFR haplotypes
    g = ds.la.global_ancestry
    np.testing.assert_allclose(g.loc[0, ["EUR", "AFR"]].to_numpy(dtype=float), [0.7, 0.3], atol=1e-6)
    # S2: valid haps EUR=4? codes -> [1,1],[1,0],[-1,0],[0,0],[1,1] -> EUR 4, AFR 5
    np.testing.assert_allclose(g.loc[1, ["EUR", "AFR"]].to_numpy(dtype=float), [4 / 9, 5 / 9], atol=1e-6)


def test_write_store_with_posterior_and_overwrite(tmp_path):
    codes = _codes()[:2]
    post = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float32)
    header = Header(samples=["S1", "S2"], ancestries=["EUR", "AFR"], source_files=[],
                    chrom="chr1", has_posterior=True,
                    global_ancestry=np.array([[0.5, 0.5], [0.5, 0.5]], np.float32))
    chunk = Chunk(np.array(["chr1"] * 2), np.array([1, 2], np.int32), np.array([1, 2], np.int32),
                  codes, posterior=post)
    path = write_store(header, [chunk], tmp_path / "chr1.zarr")
    ds = open_store(path)
    np.testing.assert_allclose(ds.la.posterior.values, post)
    with pytest.raises(FileExistsError):
        write_store(header, [chunk], path, overwrite=False)
    write_store(header, [chunk], path, overwrite=True)


def test_write_store_rejects_bad_chunk(tmp_path):
    header = Header(samples=["S1"], ancestries=["A", "B"], source_files=[], chrom="chr1")
    bad = Chunk(np.array(["chr1"]), np.array([1], np.int32), np.array([1], np.int32),
                np.zeros((1, 2, 2), np.int8))
    with pytest.raises(ValueError, match="codes have shape"):
        write_store(header, [bad], tmp_path / "x.zarr")
