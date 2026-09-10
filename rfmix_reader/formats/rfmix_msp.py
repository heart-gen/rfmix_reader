"""RFMix ``.msp.tsv`` parser (segment-level hard calls)."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import gzip
import re

import pandas as pd

from .common import align_g_anc_columns, pops_by_code
from .base import Chunk, Header, chrom_label_from_path
from .global_ancestry import frame_to_array, read_rfmix_q

FORMAT = "msp"


def _parse_pop_header(fn: str) -> Dict[str, int]:
    """
    Parse the first header line of an .msp.tsv file to extract population codes.

    Expected format:
        #Subpopulation order/codes: AFR=0   EUR=1

    Returns a dict mapping population label → integer code, e.g. {"AFR": 0, "EUR": 1}.
    """
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt") as fh:
        line = fh.readline().strip()
    pairs = re.findall(r"(\w+)=(\d+)", line)
    if not pairs:
        raise ValueError(
            f"Could not parse population codes from first line of '{fn}'. "
            f"Expected format: #Subpopulation order/codes: AFR=0 EUR=1 ..."
        )
    return {label: int(code) for label, code in pairs}


def _read_msp_file(fn: str) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    Read a single .msp.tsv file.

    Returns
    -------
    segs : DataFrame
        All segment rows with columns: chrom, spos, epos, and per-haplotype
        ancestry integer columns (Sample_1.0, Sample_1.1, …).
    hap_cols : list of str
        Names of the per-haplotype columns in the order they appear.
    pop_map : dict
        Population label → integer code from the file header.
    """
    pop_map = _parse_pop_header(fn)

    segs = pd.read_csv(
        fn, sep="\t", comment=None, skiprows=1, header=0,
        compression="infer",
    )
    segs.rename(columns={"#chm": "chrom"}, inplace=True)
    segs["chrom"] = segs["chrom"].astype("category")
    segs["spos"] = segs["spos"].astype(np.int32)
    segs["epos"] = segs["epos"].astype(np.int32)

    hap_cols = [c for c in segs.columns if c not in
                {"chrom", "spos", "epos", "sgpos", "egpos", "n snps"}]
    return segs, hap_cols, pop_map


def _sample_names_from_hap_cols(hap_cols: Sequence[str]) -> List[str]:
    if len(hap_cols) % 2 != 0:
        raise ValueError(
            f"Expected an even number of haplotype columns, got {len(hap_cols)}."
        )
    samples = []
    for i in range(0, len(hap_cols), 2):
        h0 = hap_cols[i]
        h1 = hap_cols[i + 1]
        sample0 = h0.rsplit(".", 1)[0]
        sample1 = h1.rsplit(".", 1)[0]
        if sample0 != sample1:
            raise ValueError(f"Haplotype columns are not paired: {h0}, {h1}")
        samples.append(sample0)
    return samples


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
