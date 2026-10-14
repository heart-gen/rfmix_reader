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
   ds.la.sel_chrom("chr21")                                  # contiguous slice, still lazy
   ds.la.sel_region("chr21", 15_000_000, 20_000_000)
   ds.la.locus_index("chr21", positions)                     # segment/variant index per position, -1 = none
   ds.la.counts_at("chr21", positions, method="nearest", tolerance=5_000)
   loci_df, g_anc, local_array = ds.la.to_legacy()   # the legacy (0.5) triple

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
``locus_index`` returns the same lookup as plain indices along ``variant``
(``-1`` when no segment covers the position, or beyond ``tolerance`` bp for
``nearest``); on ``ds.la.sel_chrom(chrom)`` they index that chromosome's
array directly, which is the access pattern for per-gene QTL windows: hold one
chromosome of counts in memory and slice it per gene.
``to_parquet`` writes ``<prefix>.<chrom>-<k>.parquet`` files with ``chrom``,
``pos``, ``hap`` and one int8 ``<sample>_<ancestry>`` column per pair
(sample-major), one dask block at a time.  ``interpolate`` builds a
per-chromosome variant grid (source variants plus the requested positions),
writes ``<zarr_outdir>/<chrom>/local-ancestry.zarr`` and returns a lazy
``(variant, sample, ancestry)`` DataArray; ``linear`` returns fractional
dosages between the bracketing observed loci, ``nearest`` copies the closest
observed locus, ``stepwise`` forward-fills.  Use ``nearest`` or ``stepwise``
when hard calls are wanted: rounding ``linear`` per ancestry would break the
diploid total.  ``include_source`` selects whether the source variants are
returned alongside the requested positions; they are always used as the
interpolation anchors.

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

Ancestry-specific allele frequencies (LoGicAl)
----------------------------------------------

`LoGicAl <https://github.com/student-jw/LoGicAl>`_ (Wang & Zöllner,
*Bioinformatics* 2026) estimates ancestry-specific allele frequencies from
admixed samples while modelling the uncertainty of both the local-ancestry
call and the genotype call.  Its probability-based variants take, per site and
individual, the two haplotype posteriors that only ``.fb.tsv`` carries.
``ds.la.to_logical`` aligns that posterior tensor with the phased VCF the
caller was run on and writes a per-chromosome **bundle** that the packaged R
helper turns into ``run_logical()`` calls; the estimation stays in LoGicAl.

.. code:: python

   ds = open_rfmix("rfmix_out/", source="fb", keep_posteriors=True, cache_dir="la_cache/", chrom="21")
   bundle = ds.la.to_logical("phased/chr21.vcf.gz", "logical/")       # -> logical/chr21/

.. code:: bash

   rfmix-reader export-logical la_cache/ phased/chr21.vcf.gz logical/ --chrom chr21
   Rscript logical/chr21/logical_run.R --bundle logical/chr21 --method pg --out chr21.pg.tsv

The R side needs ``arrow``, ``jsonlite``, ``SQUAREM`` and LoGicAl
(``remotes::install_github("student-jw/LoGicAl")``); ``--method phase_ag`` needs
only ``arrow`` and ``jsonlite`` and is a useful first check.  The helper's
output has one row per site: ``chrom``, ``pos``, ``id``, ``ref``, ``alt``,
``window``, ``distance``, ``method``, ``n_used`` (individuals), ``n_dropped``,
``n_haplotypes``, one ``<ancestry>_freq`` column per ancestry (for the VCF
**ALT** allele) and ``error``.

What the bundle holds (``manifest.json`` documents every field):

- the posterior tensor on its native grid (one row per RFMix CRF point or
  ``.msp`` segment, sample and haplotype: ``label`` in ``1..K``, ``0`` when the
  haplotype has no call, and ``p_<ancestry>`` when posteriors are stored);
- every biallelic VCF site mapped to one grid row (``window``) with the
  distance in bp (``nearest``, the default for ``.fb``/FLARE/haptools grids;
  ``stepwise`` for ``.msp`` segments, where the distance is 0 inside the
  segment), plus ``max_distance`` to drop sites too far from any CRF point;
- per site and sample, ``g`` (ALT dosage), ``h1``/``h2`` (ALT allele on each
  haplotype; ``-1`` for an unphased heterozygote) and, when the VCF has
  ``PL`` or ``GL``, the 4-column likelihood LoGicAl expects;
- provenance: versions, source files, the VCF's path/size/mtime, the matched
  sample set (Dataset order; samples absent from either side are listed), the
  sites skipped and why (``multiallelic``, ``no_alt``, ``non_snp``,
  ``no_window``, ``beyond_max_distance``, ``all_missing``) and a distance
  summary.

Two requirements and one convention:

- **Haplotype order.**  Haplotype 1 of the Dataset must be the first allele
  of the VCF genotype, which holds when the local ancestry was inferred from
  that same phased VCF.  A ``phase_swapped`` mask from ``ds.la.phase`` is
  applied automatically (``apply_phase_swaps=False`` to keep the VCF phase).
- **Multi-allelic records are skipped**, not split: run ``bcftools norm -m -any``
  first if they matter.
- **Missing data.**  LoGicAl has no missing-value handling, so the helper
  drops, per site, every individual with a missing input for the chosen
  method; the closed-form ``phase_ag`` is evaluated per haplotype.
  :func:`rfmix_reader.ops.logical.closed_form_frequencies` computes that same
  estimator in numpy and is the reference the R helper reproduces exactly.

The helper works around three defects of LoGicAl 0.1.0 (reported upstream):
``phase_ag`` is computed in the script; ``par0 = rep(1/K, K)`` is always passed
so the label-based methods see every ancestry; and ``SQUAREM`` is attached
because LoGicAl's NAMESPACE does not import it.

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

See ``MIGRATION.md`` in the repository: every legacy (0.5 and earlier) function maps to one
call on the Dataset, ``ds.la.to_legacy()`` returns the old triple, and
``rfmix_reader.from_legacy(loci_df, g_anc, admix)`` builds a Dataset from one.
