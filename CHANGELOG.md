# Changelog

## 0.8.0

One feature: an exporter for [LoGicAl](https://github.com/student-jw/LoGicAl)
(Wang & Zöllner, *Bioinformatics* 2026, doi:10.1093/bioinformatics/btag752),
the ancestry-specific allele-frequency estimator that models the uncertainty
of the local-ancestry call. Its probability-based variants need, per site and
individual, the two haplotype posteriors that `.fb.tsv` carries and that this
package already stores; nothing in the package consumed them downstream
before. The estimation stays in LoGicAl; the package prepares its inputs.

- `ds.la.to_logical(vcf, outdir, ...)` / `rfmix-reader export-logical` write
  a per-chromosome Parquet bundle (`io` extra): the posterior tensor on its
  native grid (CRF points or `.msp` segments; 1-based labels, `0` for a
  haplotype without a call, `p_<ancestry>` when posteriors are stored), every
  biallelic VCF site mapped to one grid row (`nearest` for point grids,
  `stepwise` for segments, with the distance and an optional `max_distance`),
  per-site genotypes (`g`, phased `h1`/`h2`, and the 4-column likelihood
  LoGicAl expects when the VCF has `PL`/`GL`), the matched sample set, the
  skipped sites with reasons, and a `manifest.json` (also stamped into every
  table's Arrow metadata). Nothing of size sites x samples x 2 x K is ever
  materialised. A `phase_swapped` mask from `ds.la.phase` is applied.
- `rfmix_reader/r/logical_run.R`, copied into every bundle, runs any LoGicAl
  method over a bundle and writes one row per site. `closed_form_frequencies`
  computes the `phase_ag` estimator in numpy and is the reference the helper
  reproduces exactly; the slow chr21 test checks the haplotype-order
  assumption against the haptools `POP` truth.
- `rfmix_reader.io.genotypes`: a cyvcf2 block reader (`iter_genotype_blocks`,
  `gt_to_calls`, `pl_to_gl4`, `gl_to_gl4`) that the exporter uses and that is
  reusable on its own.
- The helper works around three defects of LoGicAl 0.1.0, reported upstream:
  `run_logical(method = "phase_ag")` references an undefined `npop` and is
  computed in the script instead; the label-based methods take K from the
  labels present at a site when `par0` is NULL, so `par0 = rep(1/K, K)` is
  always passed; and LoGicAl's NAMESPACE does not import `SQUAREM::squarem`,
  so the helper attaches SQUAREM.

Test data: the bundled chr21 FLARE output (`data/chr21.anc.vcf.gz`, `.tbi`,
`chr21.global.anc.gz`, `chr21.model`) is regenerated. The old run called
nearly every haplotype EUR (mean AFR 0.002 against a simulated YRI fraction
of 0.789): the haptools founders (1kGP CEU and YRI) were also in FLARE's
reference panel, so every target haplotype was an exact copy of a panel
haplotype and FLARE's EM converged to `mu` = 100% EUR. The new run drops all
CEU and YRI individuals from the panel (EM on, `array=false`) and gives mean
AFR 0.794, r = 0.81 against the truth and 99.6% haplotype-level agreement
with the haptools `POP` field. A slow test now checks the FLARE fixture
against the truth, both from `global.anc.gz` and from the decoded calls.

Follow-ups, not in this release: interpolating posteriors between CRF points
(the exporter copies the nearest one; `interpolate` handles counts only) and
FLARE's `ANP1`/`ANP2` probabilities, which the FLARE parser does not keep.

## 0.7.3

`dask` was capped below 2026.0 -- a caret requirement (`^2025.1`) against a
calendar-versioned dependency, so the cap was almost certainly not meant. dask
releases monthly, which made the cap actively harmful: `pip install -U
rfmix-reader` on an environment holding dask 2026.8.0 silently downgraded it to
2025.12.0. Nothing in the package needs the cap -- the test suite passes on dask
2026.8.0 -- so the requirement is now `>=2025.1`. No code changed, and no
behaviour changes for an environment that was already within the old range.

## 0.7.2

One fix to `interpolate`, which changed its memory profile and not its output:
a chunk of the cohort was resident two and a half times over, so the peak
scaled with the sample count rather than with the machine.

- `chunk_size` is a row count, so the bytes one interpolation chunk holds grew
  with the cohort: 50,000 rows is 60 MB at 100 samples and 5.6 GB at 10,000.
  `interpolate_array` now takes `max_chunk_bytes` (2 GiB) and caps the
  effective row count by it; `interpolate` and `ds.la.interpolate` pass it
  through, and `None` honours `chunk_size` exactly, as before. Peak memory is
  now a property of the machine rather than of the cohort.
- `mod.array` -> `mod.asarray` on the chunk: `np.concatenate` already returns a
  fresh array, so the second copy of the whole chunk was pure overhead.
- The per-column NaN scan allocated one bool per element of the chunk, a
  further quarter of it (1.4 GB for the largest supported cell). It now
  reduces over locus tiles of 64 MB.
- A `(sample, ancestry)` column with no observed value anywhere cannot be
  filled by any method. Those columns are now reported once in a warning and
  kept out of the second interpolation pass, which would otherwise re-read the
  entire locus axis to return the NaNs it was given.

Profiled on a 10,000-sample chr21 grid (49,976 loci) under a 16 GB cgroup, the
interpolation phase went from an OOM kill at chunk 0 of 2 to completing in
36-47 s across runs.

Chunk-size invariance is now covered by a test and holds whenever every source
row carries a call. It does *not* hold when a source row carries a missing call
for some column: that column's chunk-local fill then depends on where the
boundaries fall. That predates this release and is unchanged by it.

## 0.7.1

Three fixes to `interpolate`, each of which changed the values it returned.

- `include_source=False` dropped the source variants from the grid handed to
  the imputer, so only requested positions that happened to coincide with a
  variant were available as anchors. A request for positions between variants
  was interpolated from whatever few anchors survived -- or left NaN when none
  did. The source variants are now always part of the interpolation grid and
  `include_source` selects only what is returned, as documented. The Zarr
  arrays on disk hold the full grid, so address the result by its
  `variant_position` coordinate rather than by row offset. `include_source`
  also no longer interpolates a chromosome with no requested positions.
- `method="linear"` rounded each ancestry column to the nearest integer. That
  is not linear interpolation and it breaks the diploid total: two bracketing
  rows summing to 2 interpolate to `(0.5, 0.5, 1.0)`, which rounded to
  `(0, 0, 1)` -- one ancestry copy for a diploid donor. `linear` now returns
  the fractional dosages, which preserve the total by construction.
  `nearest` and `stepwise` assign an observed row verbatim and remain the
  methods for hard calls.
- The interpolation position axis was float32, which cannot represent a bp
  position above ~16.7 million: the spacing is 4 bp at chr21 scale and 16 bp at
  chr1 scale, so the interpolation weights were off by a few parts per thousand
  wherever markers are dense. On a simulated chr21 cell at 2 kb marker spacing
  this moved interpolated counts by up to 0.002. The axis is now float64; it is
  one value per locus, so the wider dtype costs nothing beside the values.

## 0.7.0
- `ds.la.sel_chrom(chrom)`, `ds.la.locus_index(chrom, positions, method,
  tolerance)` and `ds.la.counts_at(...)` (also in `ops.positions`): the
  per-chromosome, index-based access that QTL mapping needs (one chromosome
  of counts in memory, a segment index per SNP). `sel_region` is now a
  contiguous slice found by binary search instead of a boolean mask
  (a 2 Mb window of an 876k-variant chromosome: ~1 s -> ~0.13 s).
- haptools parser: regions are pulled in worker processes (cyvcf2 decodes the
  ``POP`` field under the GIL, so threads gave no speed-up), the label to
  code mapping is one vectorised comparison per ancestry pair instead of a
  per-element string pipeline, and at most ``2 * n_threads`` regions are in
  flight. chr21 of a 1M-variant simulation: 122 s / 1.5 GB -> 21 s / 0.5 GB,
  identical output. Worker processes use the ``fork`` start method (never
  re-imports the caller's ``__main__``) and are skipped inside daemonic
  processes or where fork is unavailable (threads are used instead).
- ``import rfmix_reader.formats`` before ``rfmix_reader.core`` no longer
  raises a circular-import error: ``rfmix_reader.core`` resolves its reader
  and Zarr-store exports lazily, ``get_parser`` lives in ``formats.discover``.
- `counts_from_hap_codes` (and therefore `ds.la.counts`) works in the input
  dtype instead of upcasting to int64: about 4x less temporary memory per
  dask block, so materialising a whole chromosome of counts from the cache
  peaks well below the legacy reader.

## 0.6.0

Release candidate for 1.0: the rebuild is complete but the version stays
below 1.0 until CI on every supported Python and the benchmark comparison
against 0.3.1 have been reviewed. One `xarray.Dataset` schema: one `xarray.Dataset` schema, streaming parsers, a
per-chromosome Zarr cache, and Dataset operations. See `MIGRATION.md`.

### Removed
- The legacy triple API (0.5 and earlier) and its modules (`read_*`, `write_data`,
  `admix_to_bed_individual`, `generate_tagore_bed`, `extract_locus_ancestry`,
  `create_binaries`, `Chunk`, `BinaryFileNotFoundError`, `rfmix_reader.readers`,
  `rfmix_reader.utils`, the `.bin` cache, `create-binaries`).
- `phase_admix_dask_with_index` (returned counts, which phasing cannot change),
  the experimental torch HMM (`_hmm_lai`), `delete_files_or_directories`,
  cuDF/torch gating in readers.

### Changed
- All progress and diagnostics go through `logging`; nothing prints.
- `rfmix_reader.backends` exposes `use_gpu()` and `describe_gpus()`.
- Package layout: `core/` (schema, accessor, codes, zarr_io), `formats/`,
  `ops/`, `processing/`, `viz/`, `io/prepare_reference`, `cli/main`.

## 0.5.0
- Dataset operations: `ds.la.to_bed`, `at_positions`, `to_parquet`,
  `interpolate`, `to_tagore`.
- gnomix-style phasing from each sample's own posteriors (`ds.la.phase()`),
  haplotype codes preserved end to end, `phase_swapped` mask; the
  reference-panel matcher kept as `method="reference"`.
- Interpolation no longer depends on `chunk_size` (context rows per chunk).
- Legacy names run on the Dataset core with `DeprecationWarning`s.

## 0.4.0
- `open_rfmix` / `open_flare` / `open_simu` / `open_local_ancestry` /
  `convert` returning a lazy `xarray.Dataset`; streaming parsers for
  `.msp.tsv`, `.fb.tsv` (single pass, no `.bin`), FLARE and haptools VCFs;
  per-chromosome Zarr cache (`rfmix-reader convert|info`).

## 0.3.2
- Fixed silent data corruption in the `.fb.tsv` path (memmap column chunks;
  posterior truncation to int32); downstream functions accept the 3-D array;
  ancestry axis follows the tool's header order for every reader and matches
  `g_anc`; `read_simu` returns counts in genomic order; file discovery,
  packaging extras, undeclared dependencies, fixtures and CI.
