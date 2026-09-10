"""
Regression checks on the real chr21 RFMix output (git-LFS).  Run with

    pytest --run-slow tests/test_chr21_slow.py
"""
import numpy as np
import pytest

from rfmix_reader import read_rfmix_fb
from rfmix_reader.io import Chunk
from rfmix_reader.utils import get_pops


@pytest.mark.slow
def test_chr21_fb_counts_are_consistent(tmp_path, chr21_lfs):
    """Every locus/sample must carry exactly two haplotype calls (A0 + A1)."""
    loci_df, g_anc, admix, X_raw = read_rfmix_fb(
        str(chr21_lfs), binary_dir=str(tmp_path / "bin"), generate_binary=True,
        verbose=False, return_original=True, chunk=Chunk(nsamples=100, nloci=20_000),
    )
    assert list(get_pops(g_anc)) == ["AFR", "EUR"]
    assert admix.dtype == np.int8 and X_raw.dtype == np.float32
    assert admix.shape == (loci_df.shape[0], g_anc.shape[0], 2)
    assert X_raw.chunks[1] == (400, 400, 400, 400, 400)  # whole samples per block

    counts = admix.compute()
    valid = counts[:, :, 0] >= 0
    assert (counts[valid].sum(axis=1) == 2).all()
    # Sentinel rows are exactly the sample/loci where RFMix wrote an all-zero
    # haplotype (this file has ~0.7% of them).
    assert 0 < (~valid).mean() < 0.05
    rows = np.flatnonzero(~valid.all(axis=1))[:3]
    for r in rows:
        b4 = X_raw[r].compute().reshape(-1, 2, 2)
        no_mass = ~(b4 > 0).any(axis=-1)          # (samples, 2 haps)
        np.testing.assert_array_equal(~valid[r], no_mass.any(axis=1))

    # This file stores hard 0/1 posteriors; the raw matrix must be exactly that
    # for a random block that crosses a column-chunk boundary.
    block = X_raw[1000:1200, 300:500].compute()
    assert set(np.unique(block).tolist()) <= {0.0, 1.0}
    # and per haplotype, at most one population carries the mass
    b4 = X_raw[1000:1200].compute().reshape(200, -1, 2, 2)
    assert np.isin(b4.sum(axis=-1), [0.0, 1.0]).all()
