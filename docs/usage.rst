Usage
=====

Every reader returns the same lazily-evaluated :class:`xarray.Dataset`
(see :ref:`the-dataset`); ``ds.la`` provides the views and operations.

Opening data
------------

.. code:: python

   from rfmix_reader import open_rfmix, open_flare, open_simu, open_local_ancestry, convert

   ds = open_rfmix("two_pops/out/", cache_dir="la_cache/")            # .msp.tsv (default)
   ds = open_rfmix("two_pops/out/", source="fb", keep_posteriors=True,
                   cache_dir="la_cache/")                              # .fb.tsv posteriors
   ds = open_flare("flare_runs/", cache_dir="la_cache/")
   ds = open_simu("simulations/", cache_dir="la_cache/")

   ds = open_local_ancestry("la_cache/")                              # instant reopen
   ds = open_local_ancestry("la_cache/", chrom="21")
   convert("two_pops/out/", "fb", "la_cache/", keep_posteriors=True)  # cache only

``path`` is a directory, a file, or a path prefix; ``chrom=`` restricts to one
chromosome.  With ``cache_dir`` each chromosome is parsed once in a single
streaming pass into ``<cache_dir>/<chrom>.zarr`` and reopened lazily
afterwards.  Without it the Dataset is built in memory (fine for ``.msp.tsv``).

The same conversion is available on the command line::

   rfmix-reader convert fb two_pops/out/ la_cache/ --keep-posteriors
   rfmix-reader info la_cache/

.. _the-dataset:

The Dataset
-----------

.. code:: text

   dims:   variant, sample, ploidy (=2), ancestry, contig
   vars:   haplotype_ancestry (variant, sample, ploidy)            int8  code into `ancestry`, -1 missing
           posterior          (variant, sample, ploidy, ancestry)  float32  optional
           global_ancestry    (contig, sample, ancestry)           float32
   coords: chromosome, variant_position, segment_end (variant); sample_id; ancestry; contig

* Ancestries are in the tool's own order (RFMix header, FLARE ``##ANCESTRY``);
  ``global_ancestry`` uses the same order.
* ``-1`` marks a sample/locus without a call (e.g. an all-zero RFMix
  posterior); the whole count row of such a locus is ``-1``.
* Counts are derived lazily from the codes: ``ds.la.counts`` is
  ``(variant, sample, ancestry)`` int8 with values ``0/1/2``.

.. code:: python

   ds.la.counts                 # lazy dask-backed DataArray
   ds.la.haplotypes             # (variant, sample, ploidy) codes
   ds.la.posterior              # None unless keep_posteriors=True
   ds.la.global_ancestry        # DataFrame: sample_id, <ancestries>, chrom
   ds.la.samples, ds.la.ancestries, ds.la.chromosomes
   ds.la.sel_region("chr21", 15_000_000, 20_000_000)
   loci_df, g_anc, local_array = ds.la.to_legacy()   # the pre-1.0 triple

Operations
----------

.. code:: python

   import pandas as pd

   loci = pd.DataFrame({"chrom": ["chr21", "chr21"], "pos": [15_000_000, 30_000_000]})
   ds.la.at_positions(loci)                                  # per-locus haplotype counts / fractions
   ds.la.at_positions(loci, samples=["NA19700"], aggregate=False)   # per-sample copies
   ds.la.at_positions(loci, method="nearest")

   bed = ds.la.to_bed("NA19700", min_segment=3)             # constant-ancestry intervals
   tag = ds.la.to_tagore("NA19700")                          # TAGORE-annotated BED

   ds.la.to_parquet("out/", prefix="la", rows_per_file=100_000)

   variants = pd.read_parquet("genotypes/variants.parquet")  # chrom, pos
   dense = ds.la.interpolate(variants, "imputed/", method="stepwise")

``at_positions`` uses the ``[variant_position, segment_end]`` interval of each
source variant (``stepwise``) or the closest variant (``nearest``).
``to_parquet`` writes ``<prefix>.<chrom>-<k>.parquet`` files with ``chrom``,
``pos``, ``hap`` and one int8 ``<sample>_<ancestry>`` column per pair
(sample-major), one dask block at a time.  ``interpolate`` builds a
per-chromosome variant grid (source variants plus the requested positions),
writes ``<zarr_outdir>/<chrom>/local-ancestry.zarr`` and returns a lazy
``(variant, sample, ancestry)`` DataArray; ``linear`` rounds to hard calls,
``nearest`` copies the closest observed locus, ``stepwise`` forward-fills.

Phasing
-------

``ds.la.phase()`` corrects switch errors between the two haplotypes per
sample the way gnomix does, from each sample's own posteriors (or hard calls
when no posteriors are stored):

1. heterozygous blocks — runs where the two haplotypes carry different
   ancestries and the unordered pair is constant, at least
   ``min_block_len`` loci long;
2. each window of ``window_size`` loci is scored by
   ``(p0[a] + p1[b]) - (p0[b] + p1[a])`` against the block-start orientation
   ``(a, b)``; windows with ``|score| < posterior_margin`` inherit the previous
   state;
3. the two haplotypes (codes and posteriors) are exchanged wherever the state
   is "switched" — equivalent to gnomix's successive tail flips.

.. code:: python

   from rfmix_reader.processing.phase import PhasingConfig, phase_rfmix_chromosome_to_zarr, merge_phased_zarrs

   ds = open_rfmix("two_pops/out/", source="fb", keep_posteriors=True, cache_dir="la_cache/", chrom="21")
   phased = ds.la.phase(config=PhasingConfig(window_size=50, min_block_len=20, posterior_margin=0.2))
   phased["phase_swapped"].sum("variant")

   phase_rfmix_chromosome_to_zarr("two_pops/out/", None, None, "phased_chr21.zarr", chrom="21")
   merge_phased_zarrs(["phased_chr21.zarr", "phased_chr22.zarr"], "phased_all.zarr")

Phasing does not change ``ds.la.counts``; it changes which haplotype carries
which ancestry.  The previous reference-panel matcher is available as
``method="reference"`` (``ref_zarr_root`` and ``sample_annot_path`` required,
reference stores from ``prepare-reference``) for comparison only — it compares
ancestry labels against allele codes and is not a sound test of phase.

Haptools simulations
--------------------

haptools does not write chromosome lengths into the ``##contig`` header lines
but the tabix region pulls need them; reheader each VCF with the contig entry
from the ``contigs.txt`` haptools produces (``bcftools reheader``) and index
it with ``tabix`` before calling ``open_simu``.

Visualization
-------------

.. code:: python

   from rfmix_reader import plot_global_ancestry, plot_ancestry_by_chromosome, plot_local_ancestry_tagore

   g_anc = ds.la.global_ancestry
   plot_global_ancestry(g_anc, save_path="global")          # global.png + global.pdf
   plot_ancestry_by_chromosome(g_anc, save_path="by_chrom")
   plot_local_ancestry_tagore(ds.la.to_tagore("NA19700"), prefix="NA19700", build="hg38", oformat="png")

Migrating from 0.x
------------------

See ``MIGRATION.md`` in the repository: every pre-1.0 function maps to one
call on the Dataset, ``ds.la.to_legacy()`` returns the old triple, and
``rfmix_reader.from_legacy(loci_df, g_anc, admix)`` builds a Dataset from one.
