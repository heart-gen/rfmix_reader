# rfmix_reader

[![Tests](https://github.com/heart-gen/rfmix_reader/actions/workflows/tests.yml/badge.svg)](https://github.com/heart-gen/rfmix_reader/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/rfmix-reader/badge/?version=latest)](https://rfmix-reader.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/807052842.svg)](https://zenodo.org/doi/10.5281/zenodo.12629787)

Fast, memory-light access to local-ancestry output — **RFMix** (`.msp.tsv`,
`.fb.tsv`), **FLARE** (`.anc.vcf.gz`) and **haptools** simulations — as one
lazily-evaluated [`xarray.Dataset`](https://docs.xarray.dev) backed by a
per-chromosome Zarr cache.

- One streaming pass over the source (constant memory), then instant reopen.
- Haplotype ancestry codes stored as int8 (2 bytes per sample-locus, a few MB
  per chromosome on disk); posteriors kept on request.
- The same Dataset for every format, with position queries, BED intervals,
  Parquet export, interpolation onto a variant grid, and gnomix-style phasing.

## Installation

```bash
pip install rfmix-reader                 # core: numpy, pandas, dask, xarray, zarr, cyvcf2
pip install "rfmix-reader[viz,io]"       # + matplotlib/seaborn/cairosvg, pyarrow
```

| Extra | Adds | Needed for |
|---|---|---|
| `viz` | matplotlib, seaborn, cairosvg | `plot_*`, `ds.la.to_tagore` |
| `io` | pyarrow | `ds.la.to_parquet` |
| `reference` | bio2zarr | `prepare-reference` (VCF to VCF-Zarr for `method="reference"` phasing) |
| `gpu` | torch, cupy-cuda12x, cudf-cu12, dask-cudf-cu12 | optional CuPy compute backend |
| `all` | everything above | |

Parsing is CPU-only by design (it is I/O bound). When CuPy is importable the
imputation and plotting helpers use it as the array backend
(`rfmix_reader.backends.use_gpu()`).

## Quickstart

```python
from rfmix_reader import open_rfmix, open_local_ancestry

# parse once (one streaming pass), cache as la_cache/<chrom>.zarr
ds = open_rfmix("two_pops/out/", cache_dir="la_cache/")

# later sessions: instant, lazy
ds = open_local_ancestry("la_cache/")
ds = open_local_ancestry("la_cache/", chrom="21")

ds.la.counts            # (variant, sample, ancestry) int8 diploid counts, lazy dask
ds.la.haplotypes        # (variant, sample, ploidy) int8 ancestry codes
ds.la.global_ancestry   # DataFrame: sample_id, <ancestries>, chrom
ds.la.samples, ds.la.ancestries, ds.la.chromosomes
```

```bash
rfmix-reader convert fb two_pops/out/ la_cache/ --keep-posteriors
rfmix-reader info la_cache/
```

## Readers

| Source | Call | Notes |
|---|---|---|
| RFMix `.msp.tsv` (+ `.rfmix.Q`) | `open_rfmix(path)` | default; segment-level hard calls, `segment_end` coordinate |
| RFMix `.fb.tsv` (+ `.rfmix.Q`) | `open_rfmix(path, source="fb", keep_posteriors=True)` | one pass over the text; hard calls are the per-haplotype argmax, posteriors optional |
| FLARE `.anc.vcf.gz` (+ `.global.anc.gz`) | `open_flare(path)` | `AN1`/`AN2` codes in `##ANCESTRY` order |
| haptools `simgenotype --pop_field` | `open_simu(path)` | tabix regions in parallel; sorted labels (haptools defines no order) |
| existing cache | `open_local_ancestry(cache_dir, chrom=None)` | concatenates `<chrom>.zarr` stores |
| write cache only | `convert(path, fmt, cache_dir)` | `fmt` in `msp`, `fb`, `flare`, `haptools` |

`path` may be a directory, a single file, or a path prefix
(`/out/run_` matches `/out/run_chr1.msp.tsv`, ...). `chrom="21"` restricts to
one chromosome. Without `cache_dir` the Dataset is built in memory, which is
fine for `.msp.tsv`; for `.fb.tsv` with posteriors use a cache (a chr1-sized
cohort of 500 samples is ~11 GB of float32 posteriors).

## The Dataset

```text
dims:   variant, sample, ploidy (=2), ancestry, contig
vars:   haplotype_ancestry (variant, sample, ploidy)            int8  code into `ancestry`, -1 missing
        posterior          (variant, sample, ploidy, ancestry)  float32  optional
        global_ancestry    (contig, sample, ancestry)           float32
coords: chromosome, variant_position, segment_end (variant); sample_id; ancestry; contig
attrs:  source_format, source_files, rfmix_reader_version
```

Conventions that hold for every reader:

- Ancestries are in the **tool's own order** (RFMix `#reference_panel_population`
  / `#Subpopulation order/codes`, FLARE `##ANCESTRY`), and `global_ancestry`
  uses the same order.
- A sample/locus with no call (e.g. an all-zero RFMix posterior) is `-1` in the
  codes and in every count of that row.
- Counts are derived lazily from the codes (`ds.la.counts`), so they are never
  stored twice.

`ds.la` is the accessor (registered on import of `rfmix_reader.core`, which
every `open_*` call does):

| Accessor | Returns |
|---|---|
| `ds.la.counts` / `.haplotypes` / `.posterior` | lazy DataArrays (posterior may be `None`) |
| `ds.la.global_ancestry` | long DataFrame (`sample_id`, ancestries, `chrom`) |
| `ds.la.samples`, `.ancestries`, `.chromosomes`, `.n_variants`, `.n_samples` | labels / sizes |
| `ds.la.sel_region("chr21", start, end)` | Dataset subset |
| `ds.la.to_legacy()` | the pre-1.0 `(loci_df, g_anc, local_array)` triple |

## Operations

All operations are lazy or streaming; `sample` is an index or a sample ID.

| Operation | Result |
|---|---|
| `ds.la.at_positions(loci_df, method="stepwise"\|"nearest", samples=None, aggregate=True)` | ancestry at listed `chrom`/`pos` (haplotype counts and fractions, or per-sample copies with `aggregate=False`) |
| `ds.la.to_bed(sample, min_segment=1)` | constant-ancestry intervals for one sample |
| `ds.la.to_tagore(sample, palette="tab10")` | the BED annotated for TAGORE; plot with `plot_local_ancestry_tagore` |
| `ds.la.to_parquet(outdir, prefix=..., rows_per_file=...)` | `<prefix>.<chrom>-<k>.parquet` (`chrom`, `pos`, `hap`, then `<sample>_<ancestry>` int8 columns), one dask block at a time |
| `ds.la.interpolate(variants_df, zarr_outdir, method="linear"\|"nearest"\|"stepwise")` | counts on a denser variant grid, Zarr-backed per chromosome; missing calls are filled |
| `ds.la.phase(config=PhasingConfig())` | phase-corrected haplotype codes (single chromosome) |

```python
import pandas as pd

loci = pd.DataFrame({"chrom": ["chr21", "chr21"], "pos": [15_000_000, 30_000_000]})
ds.la.at_positions(loci)                       # AFR_haplotypes, AFR_fraction, ...
ds.la.at_positions(loci, samples=["NA19700"], aggregate=False)

bed = ds.la.to_bed("NA19700", min_segment=3)
ds.la.to_parquet("out/", prefix="la", rows_per_file=100_000)

variants = pd.read_parquet("genotypes/variants.parquet")   # chrom, pos
dense = ds.la.interpolate(variants, "imputed/", method="stepwise")
```

## Phasing

`ds.la.phase()` corrects switch errors between the two haplotypes per sample
the way gnomix does: heterozygous blocks are found from the haplotype codes,
each window inside a block is scored by how well the two posterior tracks match
the block-start orientation versus the swapped one, uninformative windows
inherit the previous state, and the two haplotypes (codes and posteriors) are
exchanged wherever the track says "switched". It needs no reference panel; open
`.fb.tsv` output with `keep_posteriors=True` so the posteriors are available
(without them the hard calls are used). The result adds a
`phase_swapped (variant, sample)` mask.

```python
from rfmix_reader import open_rfmix
from rfmix_reader.processing.phase import PhasingConfig, merge_phased_zarrs

ds = open_rfmix("two_pops/out/", source="fb", keep_posteriors=True,
                cache_dir="la_cache/", chrom="21")
phased = ds.la.phase(config=PhasingConfig(window_size=50, min_block_len=20, posterior_margin=0.2))
phased["phase_swapped"].sum("variant")         # exchanged loci per sample
```

Whole chromosomes phase in seconds (chr21, 175k loci by 500 samples: about
4 s). `phase_rfmix_chromosome_to_zarr(prefix, None, None, "phased_chr21.zarr", chrom="21")`
does read, phase and write in one call, and `merge_phased_zarrs` concatenates
per-chromosome stores; both outputs reopen with `open_local_ancestry`.

The previous reference-panel matcher remains as `method="reference"` (with
`ref_zarr_root` / `sample_annot_path`) for comparison only: it compares
ancestry labels against reference *allele* codes, which is not a sound test
of phase. To build its inputs, convert bgzipped, indexed reference VCFs with
`prepare-reference refs/ 1kg_chr21.vcf.gz` (needs the `reference` extra) and
give a two-column `sample_id<TAB>group` annotation table.

## Haptools simulations

haptools does **not** write the chromosome length into the `##contig` header,
but the tabix region pulls need it. Reheader each file with the contig entry
from the `contigs.txt` haptools produces, for example:

```bash
CONTIG_LINE=$(grep -w "ID=${CHR}" contigs.txt)
bcftools view -h "$IN" | sed "s/^##contig=<ID=${CHR}>.*/${CONTIG_LINE}/" > header.${CHR}.tmp
bcftools reheader -h header.${CHR}.tmp -o "$OUT" "$IN"
tabix -p vcf "$OUT"
```

## Visualization

```python
from rfmix_reader import plot_global_ancestry, plot_ancestry_by_chromosome, plot_local_ancestry_tagore

g_anc = ds.la.global_ancestry
plot_global_ancestry(g_anc, save_path="global")            # writes global.png and global.pdf
plot_ancestry_by_chromosome(g_anc, save_path="by_chrom")
plot_local_ancestry_tagore(ds.la.to_tagore("NA19700"), prefix="NA19700", build="hg38", oformat="png")
```

## Migrating from 0.x

See [MIGRATION.md](MIGRATION.md). Every pre-1.0 name maps to one call on the
Dataset; `ds.la.to_legacy()` returns the old triple and
`rfmix_reader.from_legacy(loci_df, g_anc, admix)` builds a Dataset from one.

## Development

```bash
git clone https://github.com/heart-gen/rfmix_reader.git
cd rfmix_reader
poetry install --with test --extras "viz io"
poetry run pytest             # fast suite (seconds)
poetry run pytest --run-slow  # also the chr21 tests (needs git-LFS data)
```

Test fixtures for every format live under `tests/data/` (regenerate with
`python tests/data/make_fixtures.py`).

## Citation

If you use this software, please cite:

Benjamin, K. J. M. (2024). **RFMix-reader** \[Computer software].
[https://github.com/heart-gen/rfmix\_reader](https://github.com/heart-gen/rfmix_reader)

Kynon JM Benjamin. *"RFMix-reader: Accelerated reading and processing for local ancestry studies."*
**bioRxiv** (2024).
DOI: [10.1101/2024.07.13.603370](https://www.biorxiv.org/content/10.1101/2024.07.13.603370v2).

## Funding

This work was supported by the National Institutes of Health,
National Institute on Minority Health and Health Disparities (NIMHD)
K99MD016964 / R00MD016964.
