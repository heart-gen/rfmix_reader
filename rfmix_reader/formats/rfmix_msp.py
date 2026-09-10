"""RFMix ``.msp.tsv`` parser (segment-level hard calls)."""
from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np

from ..readers._common import align_g_anc_columns, pops_by_code
from ..readers.read_msp import _read_msp_file, _sample_names_from_hap_cols
from .base import Chunk, Header, chrom_label_from_path
from .global_ancestry import frame_to_array, read_rfmix_q

FORMAT = "msp"


def discover(path: str, chrom: Optional[str] = None):
    from .discover import discover as _discover

    return _discover(path, FORMAT, chrom)


def scan(filemap: Dict[str, str], **_) -> Header:
    fn = filemap["msp.tsv"]
    segs, hap_cols, pop_map = _read_msp_file(fn)
    pops = pops_by_code(pop_map)
    samples = _sample_names_from_hap_cols(hap_cols)

    global_anc = None
    if "rfmix.Q" in filemap:
        q = align_g_anc_columns(read_rfmix_q(filemap["rfmix.Q"], add_chrom=False), pops)
        global_anc = frame_to_array(q, samples, pops)

    chrom = chrom_label_from_path(fn)
    if chrom is None and len(segs):
        chrom = str(segs["chrom"].iloc[0])
    return Header(
        samples=samples, ancestries=pops, source_files=sorted(filemap.values()),
        chrom=chrom, n_variants=int(len(segs)), global_ancestry=global_anc,
        extra={"segs": segs, "hap_cols": hap_cols},
    )


def iter_chunks(filemap: Dict[str, str], header: Header, chunk_rows: int = 10_000, **_
                ) -> Iterator[Chunk]:
    if "segs" in header.extra:
        segs, hap_cols = header.extra["segs"], header.extra["hap_cols"]
    else:
        segs, hap_cols, _ = _read_msp_file(filemap["msp.tsv"])

    n_pops = header.n_ancestries
    n_samples = header.n_samples
    hap = segs[hap_cols].to_numpy(dtype=np.int16)
    if hap.size and (hap.min() < 0 or hap.max() >= n_pops):
        raise ValueError(
            f"Ancestry codes must be in 0..{n_pops - 1} (populations "
            f"{header.ancestries}); found values in [{hap.min()}, {hap.max()}]."
        )
    codes = hap.astype(np.int8).reshape(len(segs), n_samples, 2)
    chrom = segs["chrom"].astype(str).to_numpy()
    pos = segs["spos"].to_numpy(dtype=np.int32)
    end = segs["epos"].to_numpy(dtype=np.int32)

    for start in range(0, len(segs), chunk_rows):
        stop = min(start + chunk_rows, len(segs))
        yield Chunk(chrom[start:stop], pos[start:stop], end[start:stop], codes[start:stop])
