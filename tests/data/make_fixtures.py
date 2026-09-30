"""
Generate the small committed test fixtures under ``tests/data/``.

Run once (needs ``pysam`` for bgzip + tabix):

    python tests/data/make_fixtures.py

Every fixture is tiny and deliberately uses a NON-alphabetical population
order in the tool's own header so that ordering bugs are caught.
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _bgzip_index(vcf_path: Path) -> Path:
    import pysam

    gz = Path(str(vcf_path) + ".gz")
    pysam.tabix_compress(str(vcf_path), str(gz), force=True)
    pysam.tabix_index(str(gz), preset="vcf", force=True)
    os.remove(vcf_path)
    return gz


# ---------------------------------------------------------------------------
# RFMix .msp.tsv (+ .rfmix.Q): chr1 and chr2, 3 samples, header EUR=0 AFR=1
# ---------------------------------------------------------------------------
MSP_HEADER = (
    "#Subpopulation order/codes: EUR=0\tAFR=1\n"
    "#chm\tspos\tepos\tsgpos\tegpos\tn snps\t"
    "Sample_1.0\tSample_1.1\tSample_2.0\tSample_2.1\tSample_3.0\tSample_3.1\n"
)
# (spos, epos, hap codes for S1.0 S1.1 S2.0 S2.1 S3.0 S3.1)
MSP_SEGMENTS = {
    "chr1": [
        (10000, 49999, [0, 0, 0, 1, 1, 1]),
        (50000, 89999, [0, 1, 0, 1, 1, 1]),
        (90000, 129999, [1, 1, 0, 0, 1, 0]),
        (130000, 169999, [1, 1, 0, 1, 0, 0]),
    ],
    "chr2": [
        (20000, 59999, [0, 0, 1, 1, 1, 0]),
        (60000, 99999, [0, 0, 0, 1, 1, 0]),
        (100000, 139999, [0, 0, 0, 1, 1, 1]),
        (140000, 179999, [0, 0, 1, 1, 1, 1]),
    ],
}


def make_msp() -> None:
    for chrom, segs in MSP_SEGMENTS.items():
        lines = [MSP_HEADER]
        for k, (spos, epos, codes) in enumerate(segs):
            lines.append(
                f"{chrom}\t{spos}\t{epos}\t{k * 0.1:.2f}\t{(k + 1) * 0.1:.2f}\t100\t"
                + "\t".join(str(c) for c in codes) + "\n"
            )
        _write(HERE / "msp" / f"{chrom}.msp.tsv", "".join(lines))

        # Global ancestry consistent with the segments (fraction of EUR / AFR haplotypes)
        codes = np.array([c for _, _, c in segs])  # (n_segs, 6)
        q_lines = ["#rfmix diploid global ancestry .Q format output\n", "#sample\tEUR\tAFR\n"]
        for s in range(3):
            haps = codes[:, 2 * s:2 * s + 2].ravel()
            eur = float((haps == 0).mean())
            q_lines.append(f"Sample_{s + 1}\t{eur:.5f}\t{1 - eur:.5f}\n")
        _write(HERE / "msp" / f"{chrom}.rfmix.Q", "".join(q_lines))


# ---------------------------------------------------------------------------
# RFMix .fb.tsv (+ .rfmix.Q): chr1, 4 samples x 2 pops (EUR, AFR), 5 rows,
# fractional posteriors; Sample_4 hap2 has no posterior mass on row 4.
# ---------------------------------------------------------------------------
FB_POSITIONS = [5030578, 5030960, 5031200, 5031455, 5032000]
FB_POPS = ["EUR", "AFR"]
FB_SAMPLES = ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]


def fb_posteriors() -> np.ndarray:
    """(rows, samples, 2 haps, 2 pops) float32 posteriors used by the fixture."""
    rng = np.random.default_rng(7)
    p_eur = rng.uniform(0.05, 0.95, size=(5, 4, 2)).astype(np.float32)
    post = np.stack([p_eur, 1 - p_eur], axis=-1)
    # a few clean calls
    post[0, 0, :, :] = [[1.0, 0.0], [1.0, 0.0]]     # S1 row0: EUR/EUR
    post[0, 1, :, :] = [[0.0, 1.0], [0.0, 1.0]]     # S2 row0: AFR/AFR
    post[1, 2, :, :] = [[0.51, 0.49], [0.49, 0.51]]  # S3 row1: EUR/AFR (close call)
    post[3, 3, 1, :] = [0.0, 0.0]                   # S4 row3 hap2: missing
    return np.round(post, 5).astype(np.float32)


def make_fb() -> None:
    post = fb_posteriors()
    cols = [f"{s}:::hap{h}:::{p}" for s in FB_SAMPLES for h in (1, 2) for p in FB_POPS]
    lines = [
        "#reference_panel_population:\t" + "\t".join(FB_POPS) + "\n",
        "chromosome\tphysical_position\tgenetic_position\tgenetic_marker_index\t"
        + "\t".join(cols) + "\n",
    ]
    for r, pos in enumerate(FB_POSITIONS):
        vals = post[r].reshape(-1)
        lines.append(
            f"chr1\t{pos}\t{r * 0.001:.5f}\t{r * 5}\t"
            + "\t".join(f"{v:.5f}" for v in vals) + "\n"
        )
    _write(HERE / "fb" / "chr1.fb.tsv", "".join(lines))

    q_lines = ["#rfmix diploid global ancestry .Q format output\n", "#sample\tEUR\tAFR\n"]
    for s, name in enumerate(FB_SAMPLES):
        eur = float(post[:, s, :, 0].mean())
        q_lines.append(f"{name}\t{eur:.5f}\t{1 - eur:.5f}\n")
    _write(HERE / "fb" / "chr1.rfmix.Q", "".join(q_lines))


# ---------------------------------------------------------------------------
# FLARE: chr21.anc.vcf.gz (+.tbi) with ##ANCESTRY=<EUR=0,AFR=1>, 2 samples,
# 4 variants, one missing AN1; chr21.global.anc.gz in header order.
# ---------------------------------------------------------------------------
def make_flare() -> None:
    header = (
        "##fileformat=VCFv4.2\n"
        "##source=flare.test\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">\n'
        '##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">\n'
        "##ANCESTRY=<EUR=0,AFR=1>\n"
        "##contig=<ID=chr21,length=46709983>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample_1\tSample_2\n"
    )
    records = [
        "chr21\t5030578\trs1\tC\tT\t.\tPASS\t.\tGT:AN1:AN2\t0|0:0:0\t0|1:0:1\n",
        "chr21\t5030588\trs2\tT\tC\t.\tPASS\t.\tGT:AN1:AN2\t0|0:1:0\t0|0:0:0\n",
        "chr21\t5031000\trs3\tA\tG\t.\tPASS\t.\tGT:AN1:AN2\t1|1:1:1\t0|1:.:1\n",
        "chr21\t5032000\trs4\tG\tA\t.\tPASS\t.\tGT:AN1:AN2\t0|1:0:1\t1|1:1:1\n",
    ]
    vcf = HERE / "flare" / "chr21.anc.vcf"
    _write(vcf, header + "".join(records))
    _bgzip_index(vcf)

    with gzip.open(HERE / "flare" / "chr21.global.anc.gz", "wt") as fh:
        fh.write("SAMPLE\tEUR\tAFR\nSample_1\t0.625\t0.375\nSample_2\t0.375\t0.625\n")


# ---------------------------------------------------------------------------
# haptools simgenotype: chr21.vcf.gz (+.tbi) with a POP FORMAT field, 4
# samples, 3 populations (YRI, CEU, NAT -> sorted CEU, NAT, YRI), 3 variants
# spanning two 1 Mb regions; header declares several contigs with chr21 not
# first.  Plus chr21.bp in haptools breakpoint format.
# ---------------------------------------------------------------------------
SIMU_SAMPLES = ["Sample_1", "Sample_2", "Sample_3", "Sample_4"]
SIMU_RECORDS = [
    # pos, POP per sample (hap1,hap2)
    (100, ["YRI,YRI", "CEU,YRI", "NAT,CEU", "YRI,NAT"]),
    (5000, ["YRI,CEU", "CEU,YRI", "NAT,NAT", "YRI,NAT"]),
    (1500000, ["CEU,CEU", "YRI,YRI", "NAT,CEU", "CEU,NAT"]),
]


def make_simu() -> None:
    header = (
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="All filters passed">\n'
        "##contig=<ID=chr20,length=64444167>\n"
        "##contig=<ID=chr21,length=46709983>\n"
        "##contig=<ID=chr22,length=50818468>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=POP,Number=2,Type=String,Description="Origin Population of each respective allele in GT">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(SIMU_SAMPLES) + "\n"
    )
    lines = [header]
    for pos, pops in SIMU_RECORDS:
        fields = "\t".join(f"0|1:{p}" for p in pops)
        lines.append(f"chr21\t{pos}\t21:{pos}:A:C\tA\tC\t.\tPASS\t.\tGT:POP\t{fields}\n")
    vcf = HERE / "simu" / "chr21.vcf"
    _write(vcf, "".join(lines))
    _bgzip_index(vcf)

    bp = []
    for s in SIMU_SAMPLES:
        for h in (1, 2):
            bp.append(f"{s}_{h}\n")
            bp.append("YRI\t21\t1000000\t1.5\n")
            bp.append("CEU\t21\t2000000\t3.0\n")
            bp.append("NAT\t21\t46709983\t60.0\n")
    _write(HERE / "simu" / "chr21.bp", "".join(bp))


if __name__ == "__main__":
    make_msp()
    make_fb()
    make_flare()
    make_simu()
    print("fixtures written under", HERE)
