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
