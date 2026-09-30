"""
``ds.la`` — the local-ancestry accessor registered on :class:`xarray.Dataset`.

Importing :mod:`rfmix_reader.core` (which every ``open_*`` function does)
registers the accessor.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .codes import counts_from_hap_codes
from . import schema as S

if TYPE_CHECKING:
    import dask.array as da

__all__ = ["LocalAncestryAccessor"]


def _counts_block(codes: np.ndarray, n_anc: int) -> np.ndarray:
    return counts_from_hap_codes(codes[..., 0], codes[..., 1], n_anc)


@xr.register_dataset_accessor("la")
class LocalAncestryAccessor:
    """Convenience views over a local-ancestry Dataset (see :mod:`core.schema`)."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        self._runs: Optional[List[Tuple[str, int, int, bool]]] = None

    def _chromosome_runs(self) -> List[Tuple[str, int, int, bool]]:
        """
        Contiguous runs of the ``chromosome`` coordinate, computed once per
        Dataset: ``(normalised label, start, stop, positions sorted)``.
        """
        if self._runs is None:
            from ..formats.common import normalize_chrom_label

            raw = np.asarray(self._ds[S.CHROMOSOME].values)
            runs: List[Tuple[str, int, int, bool]] = []
            if raw.size:
                change = np.flatnonzero(raw[1:] != raw[:-1]) + 1
                starts = np.concatenate([[0], change])
                stops = np.concatenate([change, [raw.size]])
                pos = np.asarray(self._ds[S.VARIANT_POSITION].values)
                for a, b in zip(starts, stops):
                    ordered = bool((np.diff(pos[a:b]) >= 0).all()) if b - a > 1 else True
                    runs.append((normalize_chrom_label(str(raw[a])), int(a), int(b), ordered))
            self._runs = runs
        return self._runs

    # ------------------------------------------------------------------ meta
    @property
    def samples(self) -> List[str]:
        return [str(s) for s in self._ds[S.SAMPLE_ID].values]

    @property
    def ancestries(self) -> List[str]:
        return [str(a) for a in self._ds[S.ANCESTRY].values]

    @property
    def chromosomes(self) -> List[str]:
        return list(dict.fromkeys(str(c) for c in self._ds[S.CHROMOSOME].values))

    @property
    def n_variants(self) -> int:
        return int(self._ds.sizes[S.VARIANT])

    @property
    def n_samples(self) -> int:
        return int(self._ds.sizes[S.SAMPLE])

    # ------------------------------------------------------------------ data
    @property
    def haplotypes(self) -> xr.DataArray:
        """``(variant, sample, ploidy)`` int8 ancestry codes, ``-1`` = missing."""
        return self._ds[S.HAPLOTYPE_ANCESTRY]

    @property
    def counts(self) -> xr.DataArray:
        """
        ``(variant, sample, ancestry)`` int8 diploid counts, derived lazily
        from the haplotype codes.  Rows with a missing haplotype are ``-1``.
        """
        import dask.array as da

        hap = da.asarray(self._ds[S.HAPLOTYPE_ANCESTRY].data)
        if len(hap.chunks[2]) != 1:
            hap = hap.rechunk({2: 2})
        n_anc = len(self.ancestries)
        out = da.map_blocks(
            _counts_block, hap, n_anc, dtype=np.int8,
            chunks=(hap.chunks[0], hap.chunks[1], (n_anc,)),
        )
        coords = {
            S.CHROMOSOME: self._ds[S.CHROMOSOME],
            S.VARIANT_POSITION: self._ds[S.VARIANT_POSITION],
            S.SAMPLE_ID: self._ds[S.SAMPLE_ID],
            S.ANCESTRY: self._ds[S.ANCESTRY],
        }
        if S.SEGMENT_END in self._ds.coords:
            coords[S.SEGMENT_END] = self._ds[S.SEGMENT_END]
        return xr.DataArray(
            out, dims=(S.VARIANT, S.SAMPLE, S.ANCESTRY), coords=coords, name="local_ancestry",
        )

    @property
    def posterior(self) -> Optional[xr.DataArray]:
        """``(variant, sample, ploidy, ancestry)`` float32 posteriors, or None."""
        return self._ds[S.POSTERIOR] if S.POSTERIOR in self._ds else None

    @property
    def global_ancestry(self) -> Optional[pd.DataFrame]:
        """
        Global ancestry as a long DataFrame: ``sample_id``, one column per
        ancestry (tool order) and ``chrom``; one block of rows per contig.
        """
        if S.GLOBAL_ANCESTRY not in self._ds:
            return None
        ga = np.asarray(self._ds[S.GLOBAL_ANCESTRY].values, dtype=np.float32)
        contigs = [str(c) for c in self._ds[S.CONTIG].values]
        frames = []
        for k, contig in enumerate(contigs):
            df = pd.DataFrame(ga[k], columns=self.ancestries)
            df.insert(0, "sample_id", self.samples)
            df["chrom"] = contig
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------- selection
    def sel_chrom(self, chrom: str) -> xr.Dataset:
        """All variants of ``chrom`` in position order (``KeyError`` if absent)."""
        from ..ops.positions import chromosome_index

        idx = chromosome_index(self._ds, chrom)
        if idx.size and idx[-1] - idx[0] + 1 == idx.size:
            return self._ds.isel({S.VARIANT: slice(int(idx[0]), int(idx[-1]) + 1)})
        return self._ds.isel({S.VARIANT: idx})

    def sel_region(self, chrom: str, start: Optional[int] = None,
                   end: Optional[int] = None) -> xr.Dataset:
        """
        Variants on ``chrom`` with ``start <= position <= end`` (inclusive);
        empty when the chromosome is absent.  A contiguous slice of the
        chromosome (binary search on the sorted positions), so only the
        touched Zarr chunks are read afterwards.
        """
        try:
            sub = self.sel_chrom(chrom)
        except KeyError:
            return self._ds.isel({S.VARIANT: slice(0, 0)})
        pos = np.asarray(sub[S.VARIANT_POSITION].values, dtype=np.int64)
        lo = 0 if start is None else int(np.searchsorted(pos, int(start), side="left"))
        hi = pos.size if end is None else int(np.searchsorted(pos, int(end), side="right"))
        return sub.isel({S.VARIANT: slice(lo, hi)})

    def locus_index(self, chrom: str, positions, *, method: str = "stepwise",
                    tolerance: Optional[int] = None) -> np.ndarray:
        """Variant index covering each position (see :func:`ops.positions.locus_index`)."""
        from ..ops.positions import locus_index

        return locus_index(self._ds, chrom, positions, method=method, tolerance=tolerance)

    def counts_at(self, chrom: str, positions, *, method: str = "stepwise",
                  tolerance: Optional[int] = None) -> np.ndarray:
        """``(n, sample, ancestry)`` int8 counts at positions (see :func:`ops.positions.counts_at`)."""
        from ..ops.positions import counts_at

        return counts_at(self._ds, chrom, positions, method=method, tolerance=tolerance)

    # ------------------------------------------------------------ operations
    def to_bed(self, sample, *, min_segment: int = 1) -> pd.DataFrame:
        """Constant-ancestry intervals for one sample (see :func:`ops.bed.to_bed`)."""
        from ..ops.bed import to_bed

        return to_bed(self._ds, sample, min_segment=min_segment)

    def to_tagore(self, sample, *, palette: str = "tab10", min_segment: int = 1) -> pd.DataFrame:
        """TAGORE-annotated BED for one sample (see :func:`ops.tagore.to_tagore`)."""
        from ..ops.tagore import to_tagore

        return to_tagore(self._ds, sample, palette=palette, min_segment=min_segment)

    def at_positions(self, loci: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Ancestry at listed positions (see :func:`ops.positions.at_positions`)."""
        from ..ops.positions import at_positions

        return at_positions(self._ds, loci, **kwargs)

    def to_parquet(self, outdir, **kwargs):
        """Stream counts to Parquet per chromosome (see :func:`ops.parquet.to_parquet`)."""
        from ..ops.parquet import to_parquet

        return to_parquet(self._ds, outdir, **kwargs)

    def interpolate(self, variants: pd.DataFrame, zarr_outdir, **kwargs) -> xr.DataArray:
        """Interpolate onto a variant grid (see :func:`ops.interpolate.interpolate`)."""
        from ..ops.interpolate import interpolate

        return interpolate(self._ds, variants, zarr_outdir, **kwargs)

    def phase(self, ref_zarr_root: Optional[str] = None, sample_annot_path: Optional[str] = None,
              **kwargs) -> xr.Dataset:
        """
        Phase-correct haplotype codes (see :func:`processing.phase.phase_dataset`).

        The default ``method="gnomix"`` needs no reference panel; it uses
        ``ds.la.posterior`` when present.
        """
        from ..processing.phase import phase_dataset

        return phase_dataset(self._ds, ref_zarr_root, sample_annot_path, **kwargs)

    # ---------------------------------------------------------------- legacy
    def to_legacy(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], "da.Array"]:
        """
        The ``(loci_df, g_anc, local_array)`` triple returned by the legacy
        ``read_*`` functions (int8 counts, tool-order ancestries).
        """
        loci_df = pd.DataFrame({
            "chromosome": pd.Categorical([str(c) for c in self._ds[S.CHROMOSOME].values]),
            "physical_position": np.asarray(self._ds[S.VARIANT_POSITION].values, dtype=np.int32),
            "i": np.arange(self.n_variants, dtype=np.int64),
        })
        return loci_df, self.global_ancestry, self.counts.data
