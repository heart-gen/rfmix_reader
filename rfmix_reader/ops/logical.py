"""
Export a local-ancestry Dataset plus a phased VCF as a LoGicAl input bundle.

LoGicAl (Wang & Zöllner, *Bioinformatics* 2026, doi:10.1093/bioinformatics/btag752)
is an R package that estimates ancestry-specific allele frequencies from
admixed samples, one site at a time, from the two haplotype ancestry
probabilities (or labels) and the genotype call or likelihood of every
individual.  Its only entry point, ``run_logical()``, takes in-memory
matrices, so this module writes a per-chromosome **Parquet bundle** that an
R helper (``logical_run.R``, copied into the bundle) turns into those calls.

The bundle keeps the posterior tensor on its native grid (RFMix CRF points, or
``.msp`` segments) and maps every genotype site to one grid row, so nothing of
size ``sites x samples x 2 x K`` is ever materialised.  Layout of
``<outdir>/<chrom>/`` (``format_version`` 1):

``manifest.json``
    Provenance and layout (see :func:`read_manifest`).
``samples.parquet``
    ``sample_idx``, ``sample_id``, ``dataset_idx``, ``vcf_idx`` -- the matched
    samples, in Dataset order.
``windows.parquet``
    ``window``, ``pos``, ``end`` -- one row per Dataset variant on the chromosome.
``posterior-<j>.parquet``
    ``window``, ``sample_idx``, ``hap``, ``label`` (1..K, 0 missing) and, when
    the Dataset carries posteriors, ``p_<ancestry>`` float32 per ancestry.
    Dense: ``row = (window - j*W) * n * 2 + sample_idx * 2 + hap``.
``sites-<k>.parquet``
    ``site``, ``pos``, ``id``, ``ref``, ``alt``, ``is_snp``, ``window``,
    ``distance``, ``n_called``, ``n_phased``, ``file``.
``genotypes-<k>.parquet``
    ``site``, ``sample_idx``, ``g``, ``h1``, ``h2`` (int8, -1 missing) and,
    when the VCF has PL/GL, ``gl_00``, ``gl_10``, ``gl_01``, ``gl_11``
    (float32, NaN missing).  Dense: ``row = (site - site_first[k]) * n + sample_idx``.
``skipped_sites.parquet``
    ``pos``, ``ref``, ``alt``, ``reason``.

Frequencies are for the VCF **ALT** allele.  Haplotype 1 of the Dataset is
assumed to be the first allele of the VCF genotype, which holds when the local
ancestry was inferred from that same phased VCF; the ``phase_swapped`` mask of
:func:`rfmix_reader.processing.phase.phase_dataset` is applied when present.
"""
from __future__ import annotations

import json
import logging
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..core import schema as S

__all__ = ["to_logical", "closed_form_frequencies", "read_manifest", "FORMAT_VERSION"]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
GL_COLUMNS = ("gl_00", "gl_10", "gl_01", "gl_11")
# Must not start with "r": R's arrow partially matches `metadata$r` against it.
MANIFEST_KEY = b"logical_bundle_manifest"
R_HELPER = Path(__file__).resolve().parent.parent / "r" / "logical_run.R"


def _import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "pyarrow is required for the LoGicAl bundle. "
            "Install it with `pip install rfmix-reader[io]`."
        ) from e
    return pa, pq


def _write_table(pq, pa, df: pd.DataFrame, path: Path, manifest_json: Optional[str],
                 row_group_size: Optional[int] = None) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if manifest_json is not None:
        meta = dict(table.schema.metadata or {})
        meta[MANIFEST_KEY] = manifest_json.encode()
        table = table.replace_schema_metadata(meta)
    pq.write_table(table, path, row_group_size=row_group_size)


def _resolve_chrom(ds: xr.Dataset, chrom: Optional[str]) -> str:
    chroms = ds.la.chromosomes
    if chrom is not None:
        return str(chrom)
    if len(chroms) == 1:
        return chroms[0]
    raise ValueError(
        f"to_logical expects a single chromosome; got {chroms}. "
        "Pass chrom=... (the CLI loops over chromosomes)."
    )


def _match_samples(ds_samples: List[str], vcf_names: List[str],
                   samples: Optional[Sequence[str]]) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
    """Matched ids in Dataset order, their Dataset and VCF indices, and the bookkeeping."""
    vcf_pos = {s: i for i, s in enumerate(vcf_names)}
    wanted = None if samples is None else {str(s) for s in samples}
    matched = [s for s in ds_samples if s in vcf_pos and (wanted is None or s in wanted)]
    if not matched:
        raise ValueError(
            "No sample of the Dataset is in the VCF"
            + ("" if wanted is None else " (after the `samples` restriction)")
            + f": Dataset has {ds_samples[:3]}..., VCF has {vcf_names[:3]}..."
        )
    ds_index = {s: i for i, s in enumerate(ds_samples)}
    dropped_ds = [s for s in ds_samples if s not in vcf_pos or (wanted is not None and s not in wanted)]
    dropped_vcf = [s for s in vcf_names if s not in ds_index or (wanted is not None and s not in wanted)]
    info = {
        "n_dataset": len(ds_samples), "n_vcf": len(vcf_names), "n_matched": len(matched),
        "n_dropped_dataset": len(dropped_ds), "n_dropped_vcf": len(dropped_vcf),
        "dropped_dataset": dropped_ds[:10], "dropped_vcf": dropped_vcf[:10],
    }
    return (matched, np.array([ds_index[s] for s in matched], dtype=np.int64),
            np.array([vcf_pos[s] for s in matched], dtype=np.int64), info)


def to_logical(
    ds: xr.Dataset, vcf_path, outdir, *, chrom: Optional[str] = None,
    samples: Optional[Sequence[str]] = None, method: Optional[str] = None,
    max_distance: Optional[int] = None, region: Optional[Tuple[int, int]] = None,
    snps_only: bool = False, sites_per_file: int = 20_000,
    windows_per_file: Optional[int] = None, apply_phase_swaps: bool = True,
    write_r_helper: bool = True, verbose: bool = False,
) -> Path:
    """
    Write the LoGicAl input bundle for one chromosome to ``<outdir>/<chrom>/``.

    Parameters
    ----------
    ds : Dataset
        A schema Dataset (``open_rfmix(..., source="fb", keep_posteriors=True)``
        for the probability-based LoGicAl variants; any Dataset for the
        label-based ones).
    vcf_path : path
        The phased VCF/BCF the local ancestry was inferred from (ALT dosage,
        phased alleles and, when present, ``PL``/``GL`` are exported).
    outdir : path
        Bundle root; the chromosome becomes a sub-directory.
    chrom : str, optional
        Required when ``ds`` holds more than one chromosome.
    samples : sequence of str, optional
        Restrict the export to these sample ids (Dataset order is kept).
    method : {"nearest", "stepwise"}, optional
        How a genotype site is mapped to a Dataset variant (see
        :func:`rfmix_reader.ops.positions.locus_index`).  Default: ``nearest``
        when the Dataset's variants are points (``.fb``, FLARE, haptools),
        ``stepwise`` when they are segments (``.msp``).
    max_distance : int, optional
        With ``nearest``, drop sites farther than this (bp) from any variant.
    region : (start, end), optional
        1-based inclusive position range to export.
    snps_only : bool
        Skip indels and other non-SNP records.
    sites_per_file : int
        Genotype sites per ``sites-<k>``/``genotypes-<k>`` file pair.  The
        genotype table of one file is held in memory (``sites_per_file x n``
        rows); lower it for very large cohorts, especially with likelihoods.
    windows_per_file : int, optional
        Dataset variants per ``posterior-<j>`` file (default: the Dataset's
        chunk length).
    apply_phase_swaps : bool
        Exchange ``h1``/``h2`` (and ``gl_10``/``gl_01``) where a
        ``phase_swapped`` mask from :func:`~rfmix_reader.processing.phase.phase_dataset`
        says the Dataset's haplotypes were swapped relative to the VCF.
    write_r_helper : bool
        Copy ``logical_run.R`` into the bundle.
    verbose : bool
        Progress bars.

    Returns
    -------
    Path
        The bundle directory ``<outdir>/<chrom>/``.
    """
    pa, pq = _import_pyarrow()
    from ..io.genotypes import iter_genotype_blocks, vcf_samples
    from ..ops.positions import locus_index

    chrom = _resolve_chrom(ds, chrom)
    sub = ds.la.sel_chrom(chrom)
    if sub.sizes[S.VARIANT] == 0:
        raise ValueError(f"Chromosome {chrom!r} has no variants.")
    ancestries = sub.la.ancestries
    K = len(ancestries)
    ds_samples = sub.la.samples
    post_da = sub.la.posterior
    has_post = post_da is not None
    source_format = str(sub.attrs.get("source_format", "unknown"))
    if not has_post and source_format == "fb":
        warnings.warn(
            "The Dataset was read from .fb.tsv without keep_posteriors=True; the bundle "
            "carries hard labels only (LoGicAl methods al / ag / phase_ag).", stacklevel=2,
        )
    swapped_da = sub["phase_swapped"] if ("phase_swapped" in sub and apply_phase_swaps) else None

    win_pos = np.asarray(sub[S.VARIANT_POSITION].values, dtype=np.int64)
    win_end = (np.asarray(sub[S.SEGMENT_END].values, dtype=np.int64)
               if S.SEGMENT_END in sub.coords else win_pos)
    if method is None:
        method = "nearest" if bool((win_end == win_pos).all()) else "stepwise"
    if method not in ("nearest", "stepwise"):
        raise ValueError("method must be 'nearest' or 'stepwise'.")
    n_windows = int(win_pos.size)

    vcf_path = Path(vcf_path)
    matched, ds_idx, vcf_idx, sample_info = _match_samples(ds_samples, vcf_samples(vcf_path), samples)
    n = len(matched)

    bundle = Path(outdir) / chrom
    bundle.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------------------------------------------------------------- samples, windows
    pd.DataFrame({
        "sample_idx": np.arange(n, dtype=np.int32), "sample_id": matched,
        "dataset_idx": ds_idx.astype(np.int32), "vcf_idx": vcf_idx.astype(np.int32),
    }).pipe(lambda df: _write_table(pq, pa, df, bundle / "samples.parquet", None))
    pd.DataFrame({
        "window": np.arange(n_windows, dtype=np.int32),
        "pos": win_pos.astype(np.int32), "end": win_end.astype(np.int32),
    }).pipe(lambda df: _write_table(pq, pa, df, bundle / "windows.parquet", None))

    # ---------------------------------------------------------------- posterior / labels
    import dask.array as da

    hap = da.asarray(sub[S.HAPLOTYPE_ANCESTRY].data)
    W = int(windows_per_file) if windows_per_file else int(hap.chunks[0][0])
    hap = hap.rechunk({0: W})
    post = da.asarray(post_da.data).rechunk({0: W}) if has_post else None
    n_blocks = hap.numblocks[0]
    iterator = range(n_blocks)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="LoGicAl posterior", unit="block")
    posterior_files: List[str] = []
    offset = 0
    n_missing_haps = 0
    for b in iterator:
        codes = np.asarray(hap.blocks[b].compute())[:, ds_idx, :]           # (w, n, 2) int8
        w = codes.shape[0]
        label = np.where(codes >= 0, codes.astype(np.int16) + 1, 0).astype(np.int8)
        n_missing_haps += int((label == 0).sum())
        cols = {
            "window": np.repeat(np.arange(offset, offset + w, dtype=np.int32), n * 2),
            "sample_idx": np.tile(np.repeat(np.arange(n, dtype=np.int32), 2), w),
            "hap": np.tile(np.array([0, 1], dtype=np.int8), w * n),
            "label": label.reshape(-1),
        }
        if has_post:
            pblock = np.asarray(post.blocks[b].compute())[:, ds_idx, :, :]   # (w, n, 2, K)
            for a, name in enumerate(ancestries):
                cols[f"p_{name}"] = np.ascontiguousarray(pblock[..., a]).reshape(-1).astype(np.float32)
        path = bundle / f"posterior-{b}.parquet"
        _write_table(pq, pa, pd.DataFrame(cols), path, None, row_group_size=n * 2 * 1000)
        posterior_files.append(path.name)
        offset += w

    # ---------------------------------------------------------------- sites / genotypes
    files_meta: List[Dict] = []
    skipped_rows: List[Tuple[int, str, str, str]] = []
    skipped_by_reason: Dict[str, int] = {}
    distances: List[np.ndarray] = []
    n_vcf_records = 0
    n_exported = 0
    n_swapped_cells = 0
    gl_source: Optional[str] = None
    site_offset = 0
    k = 0
    blocks = iter_genotype_blocks(
        vcf_path, chrom=chrom, samples=matched, region=region,
        block_size=int(sites_per_file), snps_only=snps_only,
    )
    if verbose:
        from tqdm import tqdm
        blocks = tqdm(blocks, desc="LoGicAl genotypes", unit="block")
    perm: Optional[np.ndarray] = None
    for block in blocks:
        n_vcf_records += len(block) + len(block.skipped)
        skipped_rows.extend(block.skipped)
        if perm is None:
            order = {s: i for i, s in enumerate(block.samples)}
            perm = np.array([order[s] for s in matched], dtype=np.int64)   # VCF-order -> Dataset order
        if len(block) == 0:
            continue
        g = block.g[:, perm]
        h1 = block.h1[:, perm]
        h2 = block.h2[:, perm]
        gl = block.gl[:, perm, :] if block.gl is not None else None
        if gl is not None and gl_source is None:
            gl_source = "likelihoods"

        widx = locus_index(sub, chrom, block.pos, method=method, tolerance=max_distance)
        keep = widx >= 0
        all_missing = ~(g >= 0).any(axis=1)
        for i in np.flatnonzero(~keep):
            reason = "beyond_max_distance" if (method == "nearest" and max_distance is not None) else "no_window"
            skipped_rows.append((int(block.pos[i]), str(block.ref[i]), str(block.alt[i]), reason))
        for i in np.flatnonzero(keep & all_missing):
            skipped_rows.append((int(block.pos[i]), str(block.ref[i]), str(block.alt[i]), "all_missing"))
        keep &= ~all_missing
        if not keep.any():
            continue
        sel = np.flatnonzero(keep)
        widx = widx[sel]
        pos = block.pos[sel].astype(np.int64)
        g, h1, h2 = g[sel], h1[sel], h2[sel]
        gl = gl[sel] if gl is not None else None
        dist = (np.abs(pos - win_pos[widx]) if method == "nearest"
                else np.zeros(sel.size, dtype=np.int64)).astype(np.int32)
        distances.append(dist)

        if swapped_da is not None:
            uniq, inverse = np.unique(widx, return_inverse=True)
            sw = np.asarray(swapped_da.isel({S.VARIANT: uniq}).values, dtype=bool)[:, ds_idx][inverse]
            if sw.any():
                n_swapped_cells += int(sw.sum())
                h1, h2 = np.where(sw, h2, h1), np.where(sw, h1, h2)
                if gl is not None:
                    gl = gl.copy()
                    gl[sw] = gl[sw][:, [0, 2, 1, 3]]

        m = sel.size
        site = np.arange(site_offset, site_offset + m, dtype=np.int64)
        sites = pd.DataFrame({
            "site": site, "pos": pos.astype(np.int32), "id": block.vid[sel], "ref": block.ref[sel],
            "alt": block.alt[sel], "is_snp": block.is_snp[sel], "window": widx.astype(np.int32),
            "distance": dist, "n_called": (g >= 0).sum(axis=1).astype(np.int32),
            "n_phased": ((h1 >= 0) & (h2 >= 0)).sum(axis=1).astype(np.int32),
            "file": np.full(m, k, dtype=np.int32),
        })
        geno = {
            "site": np.repeat(site, n), "sample_idx": np.tile(np.arange(n, dtype=np.int32), m),
            "g": g.reshape(-1), "h1": h1.reshape(-1), "h2": h2.reshape(-1),
        }
        if gl is not None:
            for c, name in enumerate(GL_COLUMNS):
                geno[name] = np.ascontiguousarray(gl[..., c]).reshape(-1)
        _write_table(pq, pa, sites, bundle / f"sites-{k}.parquet", None)
        _write_table(pq, pa, pd.DataFrame(geno), bundle / f"genotypes-{k}.parquet", None,
                     row_group_size=n * 1000)
        files_meta.append({
            "k": k, "n_sites": int(m), "site_first": int(site[0]), "site_last": int(site[-1]),
            "window_min": int(widx.min()), "window_max": int(widx.max()),
        })
        site_offset += m
        n_exported += m
        k += 1

    for _, _, _, reason in skipped_rows:
        skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
    _write_table(pq, pa, pd.DataFrame(skipped_rows, columns=["pos", "ref", "alt", "reason"]).astype(
        {"pos": np.int32, "ref": object, "alt": object, "reason": object}),
        bundle / "skipped_sites.parquet", None)

    if gl_source is not None:
        from cyvcf2 import VCF
        v = VCF(str(vcf_path))
        gl_source = "PL" if v.contains("PL") else "GL"
        v.close()

    d_all = np.concatenate(distances) if distances else np.zeros(0, dtype=np.int32)
    try:
        st = vcf_path.stat()
        vcf_info = {"path": str(vcf_path), "size": int(st.st_size),
                    "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
    except OSError:  # pragma: no cover
        vcf_info = {"path": str(vcf_path)}
    manifest = {
        "format_version": FORMAT_VERSION,
        "rfmix_reader_version": str(sub.attrs.get("rfmix_reader_version", S._version())),
        "created": created,
        "chromosome": chrom,
        "source_format": source_format,
        "source_files": [str(f) for f in sub.attrs.get("source_files", [])],
        "vcf": vcf_info,
        "ancestries": list(ancestries),
        "K": K,
        "has_posterior": bool(has_post),
        "gl_source": gl_source,
        "mapping": {"method": method, "max_distance": None if max_distance is None else int(max_distance),
                    "region": None if region is None else [int(region[0]), int(region[1])]},
        "samples": sample_info,
        "sites": {
            "n_vcf_records": int(n_vcf_records), "n_exported": int(n_exported),
            "skipped": skipped_by_reason,
            "distance": {
                "max": int(d_all.max()) if d_all.size else 0,
                "p50": float(np.percentile(d_all, 50)) if d_all.size else 0.0,
                "p99": float(np.percentile(d_all, 99)) if d_all.size else 0.0,
            },
        },
        "phase_swaps": {"applied": bool(swapped_da is not None), "n_swapped_cells": int(n_swapped_cells)},
        "layout": {
            "n_samples": n, "n_windows": n_windows, "sites_per_file": int(sites_per_file),
            "windows_per_file": W, "n_missing_haplotype_labels": int(n_missing_haps),
            "files": files_meta, "posterior_files": posterior_files,
        },
        "missing_encoding": "g/h1/h2 = -1; label = 0; gl_* = NaN; p_* = 0 where the haplotype has no mass",
        "allele": "frequencies are for the VCF ALT allele; h = 1 means ALT",
    }
    manifest_json = json.dumps(manifest, indent=2)
    (bundle / "manifest.json").write_text(manifest_json + "\n")
    # Stamp the manifest into every table's schema metadata.
    for p in sorted(bundle.glob("*.parquet")):
        t = pq.read_table(p)
        meta = dict(t.schema.metadata or {})
        meta[MANIFEST_KEY] = manifest_json.encode()
        pq.write_table(t.replace_schema_metadata(meta), p,
                       row_group_size=(n * 1000 if p.name.startswith("genotypes") else
                                       n * 2 * 1000 if p.name.startswith("posterior") else None))
    if write_r_helper:
        shutil.copyfile(R_HELPER, bundle / R_HELPER.name)
    logger.info("LoGicAl bundle: %d sites, %d samples, K=%d, %s -> %s",
                n_exported, n, K, "posteriors" if has_post else "labels only", bundle)
    return bundle


def read_manifest(bundle_dir) -> dict:
    """The ``manifest.json`` of a bundle directory."""
    return json.loads((Path(bundle_dir) / "manifest.json").read_text())


def _labels_for_windows(bundle: Path, manifest: dict, pq, window_min: int, window_max: int) -> Tuple[np.ndarray, int]:
    """``(window_max - window_min + 1, n, 2)`` int8 labels for a window range, plus the first window."""
    n = int(manifest["layout"]["n_samples"])
    W = int(manifest["layout"]["windows_per_file"])
    j0, j1 = window_min // W, window_max // W
    parts = []
    for j in range(j0, j1 + 1):
        t = pq.read_table(bundle / f"posterior-{j}.parquet", columns=["label"])
        parts.append(t.column("label").to_numpy().astype(np.int8))
    lab = np.concatenate(parts).reshape(-1, n, 2)
    first = j0 * W
    return lab[window_min - first: window_max - first + 1], window_min


def closed_form_frequencies(bundle_dir, *, min_haplotypes: int = 1) -> pd.DataFrame:
    """
    LoGicAl's ``phase_ag`` estimator computed in numpy from a bundle: for each
    site and ancestry, the fraction of ALT alleles among phased haplotypes
    carrying that label.  It uses only the hard labels and the phased alleles,
    so it is the reference the R helper must reproduce exactly (and the oracle
    for sample / haplotype alignment).

    Returns a DataFrame with ``chrom``, ``pos``, ``id``, ``ref``, ``alt``,
    ``window``, ``n_used`` (individuals contributing at least one haplotype),
    ``n_haplotypes`` (haplotypes with a label and a phased allele) and, per
    ancestry, ``<anc>_freq`` (NaN below ``min_haplotypes``) and ``<anc>_n``.
    """
    _, pq = _import_pyarrow()
    bundle = Path(bundle_dir)
    manifest = read_manifest(bundle)
    ancestries = manifest["ancestries"]
    K = int(manifest["K"])
    n = int(manifest["layout"]["n_samples"])
    frames = []
    for f in manifest["layout"]["files"]:
        k = f["k"]
        sites = pq.read_table(bundle / f"sites-{k}.parquet").to_pandas()
        geno = pq.read_table(bundle / f"genotypes-{k}.parquet", columns=["h1", "h2"])
        m = len(sites)
        h = np.stack([geno.column("h1").to_numpy().astype(np.int8).reshape(m, n),
                      geno.column("h2").to_numpy().astype(np.int8).reshape(m, n)], axis=-1)
        labels, first = _labels_for_windows(bundle, manifest, pq, f["window_min"], f["window_max"])
        lab = labels[sites["window"].to_numpy() - first]                     # (m, n, 2)
        valid = (lab > 0) & (h >= 0)
        out = sites[["pos", "id", "ref", "alt", "window"]].copy()
        out.insert(0, "chrom", manifest["chromosome"])
        out["n_used"] = valid.any(axis=2).sum(axis=1).astype(np.int32)
        out["n_haplotypes"] = valid.sum(axis=(1, 2)).astype(np.int32)
        for a in range(K):
            mask = valid & (lab == a + 1)
            den = mask.sum(axis=(1, 2))
            num = (mask & (h == 1)).sum(axis=(1, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                freq = np.where(den >= max(1, int(min_haplotypes)), num / den, np.nan)
            out[f"{ancestries[a]}_freq"] = freq
            out[f"{ancestries[a]}_n"] = den.astype(np.int32)
        frames.append(out)
    cols = ["chrom", "pos", "id", "ref", "alt", "window", "n_used", "n_haplotypes"] + \
           [c for a in ancestries for c in (f"{a}_freq", f"{a}_n")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
