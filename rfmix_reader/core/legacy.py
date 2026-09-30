"""
Build a Dataset from the legacy (0.5 and earlier) ``(loci_df, g_anc, local_array)`` triple.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from . import schema as S
from .codes import codes_from_counts

__all__ = ["from_legacy", "codes_from_counts"]


def from_legacy(
    loci_df: pd.DataFrame, g_anc: Optional[pd.DataFrame], admix,
    *, source_format: str = "legacy", chunk_rows: int = 10_000,
) -> xr.Dataset:
    """
    Build a Dataset from a legacy ``(loci_df, g_anc, admix)`` triple.

    ``admix`` is ``(loci, samples, ancestries)`` counts (dask or numpy).
    ``g_anc`` supplies sample IDs and ancestry labels; when it is ``None``
    samples are named ``Sample_1..N`` and ancestries ``anc_0..A-1``.
    """
    import dask.array as da

    loci_df = loci_df.to_pandas() if hasattr(loci_df, "to_pandas") else loci_df
    if g_anc is not None and hasattr(g_anc, "to_pandas"):
        g_anc = g_anc.to_pandas()

    admix = da.asarray(admix)
    if admix.ndim != 3:
        raise ValueError(f"admix must be (loci, samples, ancestries); got shape {admix.shape}.")
    L, n_samples, n_anc = admix.shape

    chrom_col = "chromosome" if "chromosome" in loci_df.columns else "chrom"
    pos_col = "physical_position" if "physical_position" in loci_df.columns else "pos"
    if chrom_col not in loci_df.columns or pos_col not in loci_df.columns:
        raise ValueError("loci_df needs chromosome/physical_position (or chrom/pos) columns.")
    if len(loci_df) != L:
        raise ValueError(f"loci_df has {len(loci_df)} rows but admix has {L} loci.")

    if g_anc is not None:
        from ..formats.common import get_pops, sample_id_list

        samples = sample_id_list(g_anc)
        pops = [str(p) for p in get_pops(g_anc)]
        if len(samples) != n_samples or len(pops) != n_anc:
            raise ValueError(
                f"g_anc implies {len(samples)} samples x {len(pops)} ancestries but admix is "
                f"{n_samples} x {n_anc}."
            )
        contigs = [str(c) for c in dict.fromkeys(g_anc["chrom"].astype(str))] if "chrom" in g_anc else None
        if contigs:
            ga = np.stack([
                g_anc.loc[g_anc["chrom"].astype(str) == c].set_index("sample_id")
                .loc[samples, pops].to_numpy(dtype=np.float32)
                for c in contigs
            ])
        else:
            ga = g_anc.set_index("sample_id").loc[samples, pops].to_numpy(dtype=np.float32)
            contigs = None
    else:
        samples = [f"Sample_{i + 1}" for i in range(n_samples)]
        pops = [f"anc_{a}" for a in range(n_anc)]
        ga, contigs = None, None

    hap = da.map_blocks(
        codes_from_counts, admix.rechunk({2: n_anc}), dtype=np.int8,
        chunks=(admix.chunks[0], admix.chunks[1], (2,)),
    )
    ds = S.build_dataset(
        loci_df[chrom_col].astype(str).to_numpy(), loci_df[pos_col].to_numpy(), None, hap,
        samples, pops, global_ancestry=ga, contig=contigs,
        source_format=source_format, chunk_rows=chunk_rows,
    )
    if ga is None:
        # derive global ancestry from the counts (fraction of valid haplotypes)
        from ..formats.global_ancestry import fractions_from_counts

        c = np.asarray(admix)
        valid = c >= 0
        totals = np.where(valid, c, 0).sum(axis=0)
        ga_arr = fractions_from_counts(totals)
        ds = ds.assign({S.GLOBAL_ANCESTRY: ((S.CONTIG, S.SAMPLE, S.ANCESTRY),
                                            np.stack([ga_arr] * ds.sizes[S.CONTIG]))})
    return ds
