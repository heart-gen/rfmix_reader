"""Stream diploid ancestry counts to Parquet files, one or more per chromosome."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..core import schema as S
from ..io._layout import flatten_names

__all__ = ["to_parquet"]


def to_parquet(
    ds: xr.Dataset, outdir, *, prefix: str = "local-ancestry",
    rows_per_file: Optional[int] = None, verbose: bool = False,
) -> List[Path]:
    """
    Write ``ds.la.counts`` as ``<outdir>/<prefix>.<chrom>-<k>.parquet``.

    Each file holds ``chrom``, ``pos``, ``hap`` (``"<chrom>_<pos>"``) and one
    int8 column per ``<sample>_<ancestry>`` in sample-major order.  Files are
    written block by block (one dask chunk in memory), so this scales to any
    chromosome size.

    Parameters
    ----------
    rows_per_file : int, optional
        Rows per output file (default: the Dataset's chunk length).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    counts = ds.la.counts.data
    if rows_per_file:
        counts = counts.rechunk({0: int(rows_per_file)})
    names = flatten_names(ds.la.samples, ds.la.ancestries)
    chrom = np.asarray(ds[S.CHROMOSOME].values).astype(str)
    pos = np.asarray(ds[S.VARIANT_POSITION].values)

    written: List[Path] = []
    file_index = {}
    offset = 0
    n_blocks = counts.numblocks[0]
    iterator = range(n_blocks)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Writing parquet", unit="block")
    for b in iterator:
        block = np.asarray(counts.blocks[b].compute())
        n = block.shape[0]
        sl = slice(offset, offset + n)
        block_chrom, block_pos = chrom[sl], pos[sl]
        offset += n
        for c in dict.fromkeys(block_chrom.tolist()):
            rows = block_chrom == c
            k = file_index.get(c, 0)
            file_index[c] = k + 1
            df = pd.DataFrame(block[rows].reshape(int(rows.sum()), -1), columns=names)
            df.insert(0, "hap", [f"{c}_{p}" for p in block_pos[rows]])
            df.insert(0, "pos", block_pos[rows])
            df.insert(0, "chrom", c)
            path = outdir / f"{prefix}.{c}-{k}.parquet"
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
            written.append(path)
    return written
