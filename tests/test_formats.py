"""Each parser: header, chunk stream, and agreement with the legacy readers."""
import numpy as np
import pandas as pd
import pytest

from rfmix_reader.core.codes import counts_from_hap_codes
from rfmix_reader.formats import discover, get_parser
from rfmix_reader.formats.base import codes_from_pairs
from rfmix_reader.formats.global_ancestry import fractions_from_counts, frame_to_array

# tests/data/flare: ##ANCESTRY=<EUR=0,AFR=1>; axis 2 == [EUR, AFR]
FLARE_EXPECTED = np.array([
    [[2, 0], [1, 1]], [[1, 1], [2, 0]], [[0, 2], [-1, -1]], [[1, 1], [0, 2]],
], dtype=np.int8)
# tests/data/simu: pops sorted CEU, NAT, YRI
SIMU_EXPECTED = np.array([
    [[0, 0, 2], [1, 0, 1], [1, 1, 0], [0, 1, 1]],
    [[1, 0, 1], [1, 0, 1], [0, 2, 0], [0, 1, 1]],
    [[2, 0, 0], [0, 0, 2], [1, 1, 0], [1, 1, 0]],
], dtype=np.int8)


def _collect(parser, filemap, header, chunk_rows, **opts):
    chunks = list(parser.iter_chunks(filemap, header, chunk_rows, **opts))
    assert all(len(c) > 0 for c in chunks)
    chrom = np.concatenate([c.chrom for c in chunks])
    pos = np.concatenate([c.pos for c in chunks])
    end = np.concatenate([c.end for c in chunks])
    codes = np.concatenate([c.codes for c in chunks])
    post = None
    if chunks[0].posterior is not None:
        post = np.concatenate([c.posterior for c in chunks])
    return chunks, chrom, pos, end, codes, post


def test_codes_from_pairs_sentinel():
    out = codes_from_pairs([0, 1, 5, -1], [1, 1, 0, 0], 2)
    np.testing.assert_array_equal(out, [[0, 1], [1, 1], [-1, 0], [-1, 0]])
    assert out.dtype == np.int8


def test_frame_to_array_and_fractions():
    df = pd.DataFrame({"sample_id": ["b", "a"], "X": [0.2, 0.9], "Y": [0.8, 0.1]})
    arr = frame_to_array(df, ["a", "b"], ["Y", "X"])
    np.testing.assert_allclose(arr, [[0.1, 0.9], [0.8, 0.2]])
    with pytest.raises(ValueError, match="lacks columns"):
        frame_to_array(df, ["a"], ["Z"])
    with pytest.raises(ValueError, match="sample"):
        frame_to_array(df, ["c"], ["X", "Y"])
    np.testing.assert_allclose(fractions_from_counts([[3, 1], [0, 0]]), [[0.75, 0.25], [0, 0]])


def test_discover_formats(msp_dir, fb_dir, flare_dir, simu_dir):
    assert [m["msp.tsv"].endswith(f"chr{c}.msp.tsv") for m, c in zip(discover(str(msp_dir), "msp"), (1, 2))] == [True, True]
    assert set(discover(str(fb_dir), "fb")[0]) == {"fb.tsv", "rfmix.Q"}
    assert set(discover(str(flare_dir), "flare")[0]) == {"anc.vcf", "global.anc"}
    assert discover(str(simu_dir), "haptools")[0]["vcf"].endswith("chr21.vcf.gz")
    assert len(discover(str(msp_dir), "msp", chrom="2")) == 1
    with pytest.raises(ValueError):
        discover(str(msp_dir), "nope")
    with pytest.raises(ValueError):
        get_parser("nope")


# --------------------------------------------------------------------------- msp
def test_msp_parser(msp_dir):
    parser = get_parser("msp")
    filemap = discover(str(msp_dir), "msp", chrom="1")[0]
    header = parser.scan(filemap)
    assert header.samples == ["Sample_1", "Sample_2", "Sample_3"]
    assert header.ancestries == ["EUR", "AFR"] and header.chrom == "chr1"
    assert header.n_variants == 4 and header.global_ancestry.shape == (3, 2)

    chunks, chrom, pos, end, codes, post = _collect(parser, filemap, header, chunk_rows=3)
    assert [len(c) for c in chunks] == [3, 1]
    assert chrom.tolist() == ["chr1"] * 4 and post is None
    assert pos.tolist() == [10000, 50000, 90000, 130000]
    assert end.tolist() == [49999, 89999, 129999, 169999]
    np.testing.assert_array_equal(codes[0], [[0, 0], [0, 1], [1, 1]])

    counts = counts_from_hap_codes(codes[..., 0], codes[..., 1], 2)
    np.testing.assert_array_equal(counts[0], [[2, 0], [1, 1], [0, 2]])
    q = pd.read_csv(msp_dir / "chr1.rfmix.Q", sep="\t", skiprows=1)
    np.testing.assert_allclose(header.global_ancestry, q[["EUR", "AFR"]].to_numpy(), atol=1e-5)


def test_msp_parser_bad_codes(tmp_path):
    (tmp_path / "chr1.msp.tsv").write_text(
        "#Subpopulation order/codes: AFR=0\tEUR=1\n#chm\tspos\tepos\tsgpos\tegpos\tn snps\tS.0\tS.1\n"
        "chr1\t1\t2\t0\t0\t1\t0\t7\n")
    parser = get_parser("msp")
    filemap = discover(str(tmp_path), "msp")[0]
    with pytest.raises(ValueError, match="codes must be"):
        list(parser.iter_chunks(filemap, parser.scan(filemap), 10))


# --------------------------------------------------------------------------- fb
def test_fb_parser(fb_dir):
    parser = get_parser("fb")
    filemap = discover(str(fb_dir), "fb")[0]
    header = parser.scan(filemap, keep_posteriors=True)
    assert header.samples == ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
    assert header.ancestries == ["EUR", "AFR"] and header.has_posterior
    assert header.n_variants is None

    chunks, chrom, pos, end, codes, post = _collect(parser, filemap, header, chunk_rows=2, keep_posteriors=True)
    assert [len(c) for c in chunks] == [2, 2, 1]
    text = pd.read_csv(fb_dir / "chr1.fb.tsv", sep="\t", skiprows=1)
    np.testing.assert_allclose(post, text.iloc[:, 4:].to_numpy(np.float32).reshape(5, 4, 2, 2), atol=1e-6)
    assert pos.tolist() == text["physical_position"].tolist() and end.tolist() == pos.tolist()
    expected = post.argmax(-1).astype(np.int8)
    expected[~(post > 0).any(-1)] = -1
    np.testing.assert_array_equal(codes, expected)
    assert codes[3, 3, 1] == -1

    # without posteriors nothing is kept
    header2 = parser.scan(filemap)
    _, _, _, _, codes2, post2 = _collect(parser, filemap, header2, chunk_rows=10)
    assert post2 is None and np.array_equal(codes2, codes)


def test_fb_parser_rejects_bad_layout(tmp_path):
    (tmp_path / "chr1.fb.tsv").write_text(
        "#reference_panel_population:\tEUR\tAFR\n"
        "chromosome\tphysical_position\tgenetic_position\tgenetic_marker_index\t"
        "S1:::hap1:::AFR\tS1:::hap1:::EUR\tS1:::hap2:::EUR\tS1:::hap2:::AFR\n")
    parser = get_parser("fb")
    with pytest.raises(ValueError, match="column layout"):
        parser.scan(discover(str(tmp_path), "fb")[0])


# --------------------------------------------------------------------------- flare
def test_flare_parser(flare_dir):
    parser = get_parser("flare")
    filemap = discover(str(flare_dir), "flare")[0]
    header = parser.scan(filemap)
    assert header.samples == ["Sample_1", "Sample_2"]
    assert header.ancestries == ["EUR", "AFR"] and header.chrom == "chr21"
    np.testing.assert_allclose(header.global_ancestry, [[0.625, 0.375], [0.375, 0.625]])

    chunks, chrom, pos, end, codes, post = _collect(parser, filemap, header, chunk_rows=3)
    assert [len(c) for c in chunks] == [3, 1]
    assert chrom.tolist() == ["chr21"] * 4
    assert pos.tolist() == [5030578, 5030588, 5031000, 5032000]
    np.testing.assert_array_equal(counts_from_hap_codes(codes[..., 0], codes[..., 1], 2), FLARE_EXPECTED)
    assert codes[2, 1, 0] == -1


# --------------------------------------------------------------------------- haptools
def test_haptools_parser(simu_dir):
    parser = get_parser("haptools")
    filemap = discover(str(simu_dir), "haptools")[0]
    header = parser.scan(filemap)
    assert header.samples == ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
    assert header.ancestries == ["CEU", "NAT", "YRI"] and header.chrom == "chr21"
    assert header.global_ancestry is None

    chunks, chrom, pos, end, codes, post = _collect(parser, filemap, header, chunk_rows=2, region_bp=1_000_000, n_threads=2)
    assert [len(c) for c in chunks] == [2, 1]
    assert chrom.tolist() == ["chr21"] * 3
    assert pos.tolist() == [100, 5000, 1500000]
    np.testing.assert_array_equal(counts_from_hap_codes(codes[..., 0], codes[..., 1], 3), SIMU_EXPECTED)


def test_haptools_parser_unknown_label_is_missing(simu_dir, tmp_path):
    parser = get_parser("haptools")
    filemap = discover(str(simu_dir), "haptools")[0]
    header = parser.scan(filemap)
    header.ancestries = ["CEU", "YRI"]      # NAT is now unknown
    _, _, _, _, codes, _ = _collect(parser, filemap, header, chunk_rows=10)
    assert codes[0, 2, 0] == -1              # Sample_3 hap1 = NAT


def test_haptools_region_pool_selection(monkeypatch):
    """Processes only with fork and a non-daemonic parent; threads otherwise."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    from rfmix_reader.formats.haptools_vcf import _region_pool

    with _region_pool(1) as pool:
        assert isinstance(pool, ThreadPoolExecutor)
    if "fork" in mp.get_all_start_methods():
        with _region_pool(2) as pool:
            assert isinstance(pool, ProcessPoolExecutor)

    class _Daemon:
        daemon = True

    monkeypatch.setattr(mp, "current_process", lambda: _Daemon())
    with _region_pool(2) as pool:
        assert isinstance(pool, ThreadPoolExecutor)


def test_haptools_from_unguarded_script_and_daemonic_worker(simu_dir, tmp_path):
    """open_simu must work from a script without an ``if __name__`` guard (no recursive
    spawning) and from inside a daemonic multiprocessing worker (no child processes)."""
    import multiprocessing as mp
    import subprocess
    import sys

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method for the daemonic-worker half")
    script = tmp_path / "unguarded.py"
    script.write_text(
        "import multiprocessing as mp, sys\n"
        "from rfmix_reader import open_simu\n"
        f"ds = open_simu({str(simu_dir)!r}, verbose=False)\n"
        "print('RAN', ds.sizes['variant'])\n"
        "def in_daemon(q):\n"
        f"    q.put(open_simu({str(simu_dir)!r}, verbose=False).sizes['variant'])\n"
        # the test's own daemon must use fork: forkserver/spawn would re-import this
        # unguarded script themselves, which is exactly what the library must not do
        "ctx = mp.get_context('fork')\n"
        "q = ctx.Queue(); p = ctx.Process(target=in_daemon, args=(q,), daemon=True); p.start()\n"
        "print('DAEMON', q.get(timeout=120)); p.join()\n"
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("RAN") == 1, proc.stdout          # top-level code ran exactly once
    assert "DAEMON 3" in proc.stdout and "RAN 3" in proc.stdout
