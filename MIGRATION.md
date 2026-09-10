# Migrating to rfmix_reader 1.0

Version 1.0 completes the rebuild around a single lazily-evaluated
`xarray.Dataset` (see the README for the schema) and removes the pre-1.0
`(loci_df, g_anc, local_array)` API that 0.5 had deprecated.

## Readers

| Before (≤ 0.5) | Now |
|---|---|
| `loci, g_anc, admix = read_rfmix(path)` | `ds = open_rfmix(path)` |
| `read_rfmix_fb(path, binary_dir=d, generate_binary=True)` | `open_rfmix(path, source="fb", cache_dir=d)` |
| `read_rfmix_fb(..., return_original=True)` → `X_raw` | `open_rfmix(..., source="fb", keep_posteriors=True)` → `ds.la.posterior` |
| `read_flare(path)` | `open_flare(path)` |
| `read_simu(path)` | `open_simu(path)` |
| `create_binaries(path, binary_dir)` / `create-binaries` | `convert(path, "fb", cache_dir)` / `rfmix-reader convert fb ...` |
| `Chunk(nsamples, nloci)` | `chunk_rows=` on `open_*` / `convert` |

The old triple is still available from any Dataset:

```python
loci_df, g_anc, local_array = ds.la.to_legacy()
```

and a Dataset can be built from a triple you already have in memory with
`rfmix_reader.from_legacy(loci_df, g_anc, admix)`.

## Downstream functions

| Before | Now |
|---|---|
| `extract_locus_ancestry(path, loci)` | `open_rfmix(path).la.at_positions(loci)` |
| `write_data(loci, g_anc, admix, outdir=..., prefix=...)` | `ds.la.to_parquet(outdir, prefix=...)` |
| `write_imputed(...)` | `ds.la.interpolate(variants, zarr_outdir)` then `to_parquet` on the result |
| `admix_to_bed_individual(loci, g_anc, admix, k)` | `ds.la.to_bed(k)` (index or sample ID) |
| `generate_tagore_bed(loci, g_anc, admix, k)` | `ds.la.to_tagore(k)` |
| `interpolate_array(variant_loci_df, admix, zarr_outdir)` | unchanged, or `ds.la.interpolate(variants, zarr_outdir)` |
| `get_pops(g_anc)`, `get_sample_names(g_anc)` | `ds.la.ancestries`, `ds.la.samples` |
| `set_gpu_environment()` | `rfmix_reader.backends.describe_gpus()` |
| `delete_files_or_directories(patterns)` | removed (use `shutil`) |

## Phasing

| Before | Now |
|---|---|
| `phase_admix_dask_with_index(admix, X_raw, ...)` (returned counts) | `ds.la.phase()` → Dataset with corrected `haplotype_ancestry` and a `phase_swapped` mask |
| `phase_rfmix_chromosome_to_zarr(prefix, ref_zarr, annot, out, binary_dir=...)` | `phase_rfmix_chromosome_to_zarr(prefix, None, None, out, chrom=...)`; reference arguments are only used with `method="reference"` |
| output store with `local_ancestry` counts | schema store readable by `open_local_ancestry` |

The default `method="gnomix"` detects switch errors from each sample's own
posteriors and needs no reference panel. The previous reference-panel matcher
remains as `method="reference"` for comparison only.

## Data conventions (since 0.3.2)

- Counts are `int8` with `-1` for a missing call; the `.fb.tsv` hard calls are
  the per-haplotype argmax of the posteriors.
- Axis order of ancestries is the tool's own header order for every reader and
  matches `ds.la.ancestries` / the `g_anc` columns.
- Parquet columns are sample-major (`S1_EUR, S1_AFR, S2_EUR, ...`).

## Removed modules

`rfmix_reader.readers`, `rfmix_reader.utils`, `rfmix_reader.io.write_data`,
`rfmix_reader.io.loci_bed`, `rfmix_reader.io.chunk`, `rfmix_reader.io.errors`,
`rfmix_reader.processing._hmm_lai`, `rfmix_reader._testing`, and the
`create-binaries` console script.  Accessing a removed top-level name raises an
`AttributeError` that names the replacement.
