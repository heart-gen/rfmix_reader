import pytest

np = pytest.importorskip("numpy")
from rfmix_reader.processing import phase


def test_find_heterozygous_blocks_respects_min_length():
    hap0 = np.array([0, 0, 1, 1, 0])
    hap1 = np.array([0, 1, 1, 0, 0])

    blocks = phase.find_heterozygous_blocks(hap0, hap1, min_block_len=2)

    assert len(blocks) == 1
    blk = blocks[0]
    assert blk.start == 1 and blk.stop == 4


def test_assign_reference_per_window_marks_ambiguous_windows():
    hap = np.array([0, 0, 1, 1])
    refs = np.array([
        [0, 0, 0, 0],  # matches first window
        [1, 1, 1, 1],  # matches second window
    ])

    ref_track = phase.assign_reference_per_window(
        hap=hap, refs=refs, window_size=2, max_mismatch_frac=0.5
    )

    assert np.array_equal(ref_track, np.array([1, 2], dtype=np.int8))

    # If all references exceed the mismatch threshold, window should be 0
    bad_refs = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    uninformative = phase.assign_reference_per_window(
        hap=hap, refs=bad_refs, window_size=2, max_mismatch_frac=0.25
    )
    assert np.array_equal(uninformative, np.array([0, 0], dtype=np.int8))


def test_build_phase_track_from_ref_ignores_ambiguous_regions():
    ref_track = np.array([0, 1, 1, 0, 2, 2], dtype=np.int8)
    expected = np.array([0, 0, 0, 0, 1, 1], dtype=np.int8)

    phase_track = phase.build_phase_track_from_ref(ref_track)

    assert np.array_equal(phase_track, expected)


def test_apply_phase_track_swaps_tail_on_change_points():
    hap0 = np.array([0, 0, 1, 1])
    hap1 = np.array([1, 1, 0, 0])
    phase_track = np.array([0, 0, 1, 1], dtype=np.int8)

    corr0, corr1 = phase.apply_phase_track(hap0, hap1, phase_track, window_size=1)

    assert np.array_equal(corr0, np.array([0, 0, 0, 0]))
    assert np.array_equal(corr1, np.array([1, 1, 1, 1]))


def test_phase_local_ancestry_sample_phases_single_block():
    hap0 = np.array([0, 0, 1, 1])
    hap1 = np.array([1, 1, 0, 0])
    refs = np.array([
        [0, 0, 0, 0],  # matches first two sites
        [1, 1, 1, 1],  # matches last two sites
    ])

    config = phase.PhasingConfig(window_size=2, min_block_len=1)
    h0_corr, h1_corr = phase.phase_local_ancestry_sample(hap0, hap1, refs, config)

    assert np.array_equal(h0_corr, np.array([0, 0, 0, 0]))
    assert np.array_equal(h1_corr, np.array([1, 1, 1, 1]))


# ---------------------------------------------------------------------------
# Dataset-based phasing (haplotype codes in, haplotype codes out)
# ---------------------------------------------------------------------------

def _synthetic_refs(n_anc: int, L: int):
    """One 'reference' per ancestry whose code equals the ancestry index."""
    return np.repeat(np.arange(n_anc, dtype=np.int8)[:, None], L, axis=1)


@pytest.fixture
def fake_refs(monkeypatch):
    def builder(zarr_root, annot_path, chrom, positions, groups=None, hap_index_in_zarr=0, **kw):
        refs = _synthetic_refs(2, len(positions))
        return refs, ["A", "B"], {"matched_fraction": 1.0}
    monkeypatch.setattr(phase, "build_reference_haplotypes_from_zarr", builder)
    return builder


def test_phase_haplotypes_flips_tail_and_keeps_counts(fake_refs):
    import dask.array as da

    # sample 0: a switch error half-way through a heterozygous block
    hap = np.zeros((8, 2, 2), dtype=np.int8)
    hap[:, 0, 0] = [0, 0, 0, 0, 1, 1, 1, 1]
    hap[:, 0, 1] = [1, 1, 1, 1, 0, 0, 0, 0]
    hap[:, 1, :] = -1                                  # missing sample passes through
    config = phase.PhasingConfig(window_size=2, min_block_len=1)
    out = phase.phase_haplotypes(da.from_array(hap), np.arange(8) * 10, "chr1", "ref", "annot",
                                 config=config, method="reference")
    res = out.compute()
    assert res.dtype == np.int8 and out.chunks[1] == (1, 1)
    np.testing.assert_array_equal(res[:, 0, 0], 0)
    np.testing.assert_array_equal(res[:, 0, 1], 1)
    np.testing.assert_array_equal(res[:, 1, :], -1)
    # diploid counts are unchanged by phasing
    from rfmix_reader.readers._common import counts_from_hap_codes
    np.testing.assert_array_equal(counts_from_hap_codes(res[..., 0], res[..., 1], 2),
                                  counts_from_hap_codes(hap[..., 0], hap[..., 1], 2))


def test_phase_dataset_and_zarr_round_trip(fake_refs, msp_dir, tmp_path):
    from rfmix_reader import open_local_ancestry, open_rfmix
    from rfmix_reader.core import validate

    ds = open_rfmix(str(msp_dir), chrom="1", verbose=False)
    config = phase.PhasingConfig(window_size=1, min_block_len=1)
    phased = phase.phase_dataset(ds, "ref", "annot", config=config, method="reference")
    validate(phased)
    assert phased.attrs["phased"] is True and phased.attrs["phasing_method"] == "reference"
    assert "phase_swapped" in phased
    np.testing.assert_array_equal(phased.la.counts.values, ds.la.counts.values)

    with pytest.raises(ValueError, match="single chromosome"):
        phase.phase_dataset(open_rfmix(str(msp_dir), verbose=False), "ref", "annot", method="reference")

    out = phase.phase_rfmix_chromosome_to_zarr(str(msp_dir), "ref", "annot",
                                               str(tmp_path / "chr1.zarr"), chrom="1",
                                               source="msp", config=config, verbose=False,
                                               method="reference")
    reopened = open_local_ancestry(tmp_path)
    np.testing.assert_array_equal(reopened.haplotype_ancestry.values, out.haplotype_ancestry.values)
    assert reopened.la.samples == ds.la.samples and reopened.la.chromosomes == ["chr1"]

    phase.phase_rfmix_chromosome_to_zarr(str(msp_dir), "ref", "annot", str(tmp_path / "m" / "chr2.zarr"),
                                         chrom="2", source="msp", config=config, verbose=False,
                                         method="reference")
    merged = phase.merge_phased_zarrs([str(tmp_path / "m" / "chr2.zarr"), str(tmp_path / "chr1.zarr")],
                                      str(tmp_path / "merged.zarr"))
    assert merged.la.chromosomes == ["chr1", "chr2"] and merged.sizes["variant"] == 8
    assert open_local_ancestry(tmp_path, chrom="1").sizes["variant"] == 4


def test_phase_admix_dask_with_index_is_deprecated(fake_refs):
    import dask.array as da

    X = np.array([[1, 0, 0, 1], [0.9, 0.1, 0.2, 0.8]], dtype=np.float32)   # 1 sample, 2 pops
    admix = da.from_array(np.array([[[1, 1]], [[1, 1]]], dtype=np.int8))
    with pytest.warns(DeprecationWarning, match="phase_haplotypes"):
        out = phase.phase_admix_dask_with_index(admix, da.from_array(X), np.array([1, 2]), "chr1",
                                                "ref", "annot", phase.PhasingConfig(1, 1))
    np.testing.assert_array_equal(out.compute(), admix.compute())


# ---------------------------------------------------------------------------
# gnomix method: switch errors detected from the haplotypes' own posteriors
# ---------------------------------------------------------------------------

def _switch_case(L=40, block=(10, 30), switch=20, conf=0.9):
    """Sample with one het block (EUR/AFR) carrying a switch error at `switch`."""
    hap0 = np.zeros(L, dtype=np.int8)
    hap1 = np.zeros(L, dtype=np.int8)
    b0, b1 = block
    hap0[b0:switch], hap1[b0:switch] = 0, 1
    hap0[switch:b1], hap1[switch:b1] = 1, 0
    eye = np.eye(2, dtype=np.float32)
    post0 = eye[hap0] * conf + (1 - conf) * (1 - eye[hap0])
    post1 = eye[hap1] * conf + (1 - conf) * (1 - eye[hap1])
    return hap0, hap1, post0, post1


def test_gnomix_mask_detects_switch_and_apply_restores_phase():
    hap0, hap1, post0, post1 = _switch_case()
    cfg = phase.PhasingConfig(window_size=5, min_block_len=5)
    sw = phase.gnomix_switch_mask_sample(hap0, hap1, post0, post1, cfg)
    expected = np.zeros(40, dtype=bool)
    expected[20:30] = True
    np.testing.assert_array_equal(sw, expected)

    # hard-call variant (no posteriors) finds the same switch
    np.testing.assert_array_equal(phase.gnomix_switch_mask_sample(hap0, hap1, None, None, cfg), expected)

    # applying the swaps restores a constant orientation inside the block
    h0c = np.where(sw, hap1, hap0)
    h1c = np.where(sw, hap0, hap1)
    assert (h0c[10:30] == 0).all() and (h1c[10:30] == 1).all()


def test_gnomix_mask_uninformative_windows_inherit_state():
    hap0, hap1, post0, post1 = _switch_case()
    post0[25:30] = 0.5                      # no evidence either way after the switch
    post1[25:30] = 0.5
    cfg = phase.PhasingConfig(window_size=5, min_block_len=5, posterior_margin=0.2)
    sw = phase.gnomix_switch_mask_sample(hap0, hap1, post0, post1, cfg)
    assert sw[20:30].all() and not sw[:20].any() and not sw[30:].any()


def test_gnomix_mask_ignores_weak_evidence_and_short_blocks():
    # no switch in the codes; posteriors lean slightly towards a swap in one window
    hap0, hap1, post0, post1 = _switch_case(switch=30)
    post0[15:20] = [[0.45, 0.55]] * 5       # hap0 says EUR but leans AFR
    post1[15:20] = [[0.55, 0.45]] * 5
    cfg = phase.PhasingConfig(window_size=5, min_block_len=5, posterior_margin=0.5)
    assert not phase.gnomix_switch_mask_sample(hap0, hap1, post0, post1, cfg).any()
    # with a small margin that window is flipped in and, on the strong evidence
    # that follows, flipped back out (two switch points)
    cfg_small = phase.PhasingConfig(window_size=5, min_block_len=5, posterior_margin=0.1)
    sw = phase.gnomix_switch_mask_sample(hap0, hap1, post0, post1, cfg_small)
    assert sw[15:20].all() and not sw[:15].any() and not sw[20:].any()

    # a block shorter than min_block_len is left alone
    hap0, hap1, post0, post1 = _switch_case(block=(10, 16), switch=13)
    cfg = phase.PhasingConfig(window_size=1, min_block_len=10)
    assert not phase.gnomix_switch_mask_sample(hap0, hap1, post0, post1, cfg).any()


def test_gnomix_mask_missing_homozygous_and_pair_changes():
    hap0 = np.array([0, 0, -1, -1, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2], dtype=np.int8)
    hap1 = np.array([0, 0, -1, -1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0], dtype=np.int8)
    cfg = phase.PhasingConfig(window_size=1, min_block_len=1)
    sw = phase.gnomix_switch_mask_sample(hap0, hap1, None, None, cfg)
    # block {0,1} at 4..7 starts (0,1): loci 5,6 are swapped; block {0,2} at 8..13
    # starts (0,2): loci 9,10 and 13 are swapped; missing/homozygous untouched
    expected = np.zeros(14, dtype=bool)
    expected[[5, 6, 9, 10, 13]] = True
    np.testing.assert_array_equal(sw, expected)
    assert phase.gnomix_switch_mask_sample(np.array([], dtype=np.int8), np.array([], dtype=np.int8),
                                           None, None, cfg).size == 0


def test_phase_dataset_gnomix_with_posteriors(tmp_path):
    from rfmix_reader.core import build_dataset, validate

    hap0, hap1, post0, post1 = _switch_case()
    hap = np.stack([np.stack([hap0, hap1], -1), np.zeros((40, 2), np.int8)], axis=1)  # 2 samples
    post = np.zeros((40, 2, 2, 2), dtype=np.float32)
    post[:, 0, 0, :] = post0
    post[:, 0, 1, :] = post1
    post[:, 1, :, 0] = 1.0
    ds = build_dataset(["chr1"] * 40, np.arange(40) * 100, None, hap, ["S1", "S2"], ["EUR", "AFR"],
                       posterior=post, global_ancestry=np.full((2, 2), 0.5, np.float32))
    cfg = phase.PhasingConfig(window_size=5, min_block_len=5)
    out = ds.la.phase(config=cfg)
    validate(out)
    assert out.attrs["phasing_method"] == "gnomix"
    sw = out["phase_swapped"].values
    assert sw[20:30, 0].all() and sw[:, 1].sum() == 0 and sw[:20, 0].sum() == 0
    codes = out.haplotype_ancestry.values
    assert (codes[10:30, 0, 0] == 0).all() and (codes[10:30, 0, 1] == 1).all()
    np.testing.assert_array_equal(out.la.counts.values, ds.la.counts.values)
    # posteriors travel with their haplotype
    np.testing.assert_allclose(out.posterior.values[20:30, 0, 0, :], post1[20:30])
    np.testing.assert_allclose(out.posterior.values[20:30, 0, 1, :], post0[20:30])
    # dask output is lazy
    assert hasattr(out.haplotype_ancestry.data, "dask")


def test_phase_rfmix_chromosome_to_zarr_gnomix_fb(fb_dir, tmp_path):
    from rfmix_reader import open_local_ancestry

    cfg = phase.PhasingConfig(window_size=1, min_block_len=1)
    out = phase.phase_rfmix_chromosome_to_zarr(str(fb_dir), None, None, str(tmp_path / "chr1.zarr"),
                                               chrom="1", config=cfg, verbose=False)
    assert out.attrs["phasing_method"] == "gnomix" and "posterior" in out
    re = open_local_ancestry(tmp_path)
    assert "phase_swapped" in re and re.la.posterior is not None
    np.testing.assert_array_equal(re.haplotype_ancestry.values, out.haplotype_ancestry.values)


def test_phase_haplotypes_requires_reference_args():
    import dask.array as da

    hap = da.zeros((4, 1, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="positions"):
        phase.phase_haplotypes(hap, method="reference")
    with pytest.raises(ValueError, match="ref_zarr_root"):
        phase.phase_haplotypes(hap, np.arange(4), "chr1", method="reference")
    with pytest.raises(ValueError, match="method"):
        phase.phase_haplotypes(hap, method="nope")
