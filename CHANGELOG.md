# Changelog

## Unreleased
- haptools parser: regions are pulled in worker processes (cyvcf2 decodes the
  ``POP`` field under the GIL, so threads gave no speed-up), the label to
  code mapping is one vectorised comparison per ancestry pair instead of a
  per-element string pipeline, and at most ``2 * n_threads`` regions are in
  flight. chr21 of a 1M-variant simulation: 122 s / 1.5 GB -> 21 s / 0.5 GB,
  identical output.
- ``import rfmix_reader.formats`` before ``rfmix_reader.core`` no longer
  raises a circular-import error (``get_parser`` moved to
  ``formats.discover``; ``formats.base`` no longer imports the core package).
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
