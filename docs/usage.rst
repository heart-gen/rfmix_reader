Usage
=====

``rfmix-reader`` mirrors the API style of `pandas-plink` while providing
specialized helpers for the main sources of local ancestry results
(RFMix, FLARE, and Haptools simulations). The sections below show how to
prepare inputs, load the supported formats, interpret the returned
objects, and build quick visualizations.

Preparing input data
--------------------

The most common workflow starts from the RFMix ``*.fb.tsv`` files. These
can be converted into compact binary chunks with
``rfmix_reader.create_binaries``.

.. code:: python

   from rfmix_reader import create_binaries

   prefix_path = "../examples/two_populations/out/"
   create_binaries(prefix_path)

::

   Created binary files at: ./binary_files
   Converting fb files to binary!
   100% 3/3 [05:03<00:00, 101.27s/it]
   Successfully converted 3 files to binary format.

As of **v0.1.20** the same conversion can be launched via the CLI.

.. code:: shell

   create-binaries -h

Preparing reference stores for phasing
--------------------------------------

The phasing path in ``read_rfmix`` expects per-chromosome VCF-Zarr stores and
sample annotations. The ``prepare-reference`` CLI converts bgzipped, indexed
VCF/BCF files into the required ``<chrom>.zarr`` directories.

.. code:: shell

   prepare-reference -h

.. note::

   .. code-block:: text

      usage: prepare-reference [-h] [--chunk-length CHUNK_LENGTH]
                               [--samples-chunk-size SAMPLES_CHUNK_SIZE]
                               [--worker-processes WORKER_PROCESSES]
                               [--verbose | --no-verbose] [--version]
                               output_dir vcf_paths [vcf_paths ...]

      Convert one or more bgzipped reference VCF/BCF files into Zarr stores.

      positional arguments:
        output_dir            Directory where the Zarr outputs will be written.
        vcf_paths             Paths to reference VCF/BCF files (bgzipped and indexed).

      options:
        -h, --help            show this help message and exit
        --chunk-length CHUNK_LENGTH
                              Genomic chunk size for the output Zarr stores (default: 100000).
        --samples-chunk-size SAMPLES_CHUNK_SIZE
                              Chunk size for samples in the output Zarr stores (default: library chosen).
        --worker-processes WORKER_PROCESSES
                              Number of worker processes to use for conversion (default: 0, use library default).
        --verbose, --no-verbose
                              Print progress messages (default: enabled).
        --version             Show the version of the program and exit.

To build a phasing-ready reference set end-to-end::

   # sample annotations: sample_id<TAB>group (no header)
   cat > sample_annotations.tsv <<'EOF'
   NA19700	AFR
   NA19701	AFR
   NA20847	EUR
   EOF

   # generate per-chromosome VCF-Zarr stores
   prepare-reference refs/ 1kg_chr20.vcf.gz 1kg_chr21.vcf.gz \
     --chunk-length 50000 --samples-chunk-size 512

   # phase each chromosome and write to Zarr
   phase_rfmix_chromosome_to_zarr(
       file_prefix="../examples/two_populations/out/",
       ref_zarr_root="refs",
       sample_annot_path="sample_annotations.tsv",
       output_path="./phased_chr21.zarr",
       chrom="21",
   )

.. note::

   .. code-block:: text

      usage: create-binaries [-h] [--version] [--binary_dir BINARY_DIR] file_prefix

      Create binary files from RFMix *.fb.tsv files.

      positional arguments:
        file_prefix           The prefix used to identify the relevant FB TSV files.

      options:
        -h, --help            show this help message and exit
        --version             Show the version of the program and exit.
        --binary_dir BINARY_DIR
                              The directory where the binary files will be stored.
                              Defaults to './binary_files'.

Reading RFMix output
--------------------

``read_rfmix`` combines the metadata, global ancestry, and local ancestry
files that RFMix produces per chromosome. It optionally creates missing
binary files on the fly so you can point the function directly at the
RFMix output directory.

.. code:: python

   from rfmix_reader import read_rfmix

   loci_df, g_anc, admix = read_rfmix(
       "../examples/two_populations/out/",
       binary_dir="../examples/two_populations/out/binary_files",
   )

To restrict the reader to a single chromosome (instead of concatenating
everything under ``file_prefix``), pass the ``chrom`` keyword:

.. code:: python

   loci_df, g_anc, admix = read_rfmix(
       "../examples/two_populations/out/",
       chrom="21",
   )

Phasing now lives outside :func:`read_rfmix`. Use
:func:`rfmix_reader.processing.phase.phase_rfmix_chromosome_to_zarr` to
phase one chromosome at a time and
:func:`rfmix_reader.processing.phase.merge_phased_zarrs` to stitch the
results together:

.. code:: python

   from rfmix_reader.processing.phase import (
       phase_rfmix_chromosome_to_zarr,
       merge_phased_zarrs,
   )

   # Phase a single chromosome into Zarr
   ds = phase_rfmix_chromosome_to_zarr(
       file_prefix="../examples/two_populations/out/",
       ref_zarr_root="./reference_zarr",
       sample_annot_path="sample_annotations.tsv",
       output_path="./phased_chr21.zarr",
       chrom="21",
   )

   # Merge per-chromosome Zarr outputs later on
   merged = merge_phased_zarrs(
       ["./phased_chr21.zarr", "./phased_chr22.zarr"],
       output_path="./phased_all.zarr",
   )

From the command line, the same merge can be performed with::

   merge-phased-zarrs ./phased_all.zarr ./phased_chr21.zarr ./phased_chr22.zarr

Step-by-step phasing tutorial
-----------------------------

The phasing helpers wrap the RFMix VCF and sample annotations so you can
align haplotypes across chromosomes before downstream analysis. FLARE
outputs are already phased and should **skip** this entire section.

1. **Prepare phasing inputs**

   - Confirm that each RFMix ``*.fb.tsv`` file has a matching binary copy
     in ``binary_dir``. If not, generate them with
     ``rfmix_reader.create_binaries`` or the ``create-binaries`` CLI
     shown above.
   - Convert phased reference VCFs to Zarr with ``prepare-reference`` to
     enable fast random access during phasing.
   - Provide the sample annotation TSV expected by RFMix, typically
     containing ``ID`` and population columns.

2. **Run the per-chromosome phasing pipeline**

   .. code:: python

      from rfmix_reader.processing.phase import phase_rfmix_chromosome_to_zarr

      ds = phase_rfmix_chromosome_to_zarr(
          file_prefix="../examples/two_populations/out/",
          ref_zarr_root="./reference_zarr",
          sample_annot_path="sample_annotations.tsv",
          output_path="./phased_chr21.zarr",
          chrom="21",
      )

3. **Merge Zarr stores (optional)**

   .. code:: python

      from rfmix_reader.processing.phase import merge_phased_zarrs

      merged = merge_phased_zarrs(
          ["./phased_chr21.zarr", "./phased_chr22.zarr"],
          output_path="./phased_all.zarr",
      )

Reading FLARE output
--------------------

``read_flare`` offers the same interface but targets FLARE's
``*.anc.vcf.gz`` and ``global.anc.gz`` files. The output tuple mirrors
``read_rfmix`` so downstream code can remain identical.

.. code:: python

   from rfmix_reader import read_flare

   loci_df, g_anc, admix = read_flare("/path/to/flare/prefix/")

Reading Haptools simulations
----------------------------

``read_simu`` consumes BGZF-compressed VCF files generated by
``haptools simgenotype --pop_field``. It infers the list of chromosomes
in the directory, computes the global ancestry proportions from the
embedded ``POP`` field, and builds the same ``(loci_df, g_anc, admix)``
structure that ``read_rfmix`` returns.

.. code:: python

   from rfmix_reader import read_simu

   loci_df, g_anc, admix = read_simu("/path/to/simulations/")

.. note::

   Haptools does **not** emit the chromosome length inside the
   ``##contig`` header line, but ``read_simu`` requires that information
   to index each BGZF-compressed VCF file. You can copy the
   ``contigs.txt`` file that Haptools creates from the reference FASTA
   and use it to reheader every simulated chromosome before calling
   ``read_simu``. The following shell snippet shows one way to do this
   with ``bcftools`` and ``tabix``::

      CONTIGS="../../three_populations/_m/contigs.txt"
      VCFDIR="gt-files"
      CHR="chr${SLURM_ARRAY_TASK_ID}"
      OUT="${VCFDIR}/${CHR}.vcf.gz"
      IN="${VCFDIR}/back/${CHR}.vcf.gz"

      CONTIG_LINE=$(grep -w "ID=${CHR}" "$CONTIGS")
      if [[ -z "$CONTIG_LINE" ]]; then
          echo "ERROR: No contig line found for ${CHR} in $CONTIGS"
          exit 1
      fi

      bcftools view -h "$IN" \
          | sed "s/^##contig=<ID=${CHR}>.*/${CONTIG_LINE}/" > header.${CHR}.tmp
      bcftools reheader -h header.${CHR}.tmp -o "$OUT" "$IN"
      tabix -p vcf "$OUT"

Understanding the outputs
=========================

Regardless of the reader that produced them, the return values are
consistent and designed to mimic the ``pandas-plink`` conventions.

``loci_df``
-----------

``loci_df`` stores per-locus metadata.

.. code:: python

   loci_df.shape

::

   (646287, 3)

.. code:: python

   loci_df.head()

::

          chromosome  physical_position  i
   0           chr20              60137  0
   1           chr20              60291  1
   2           chr20              60340  2
   3           chr20              60440  3
   4           chr20              60823  4

The ``i`` column mirrors ``pandas_plink`` and simplifies merges with
other variant-level resources.

``g_anc``
---------

``g_anc`` (historically called ``rf_q``) contains the per-chromosome
global ancestry proportions for each individual.

.. code:: python

   g_anc.shape

::

   (1500, 4)

.. code:: python

   g_anc.head()

::

          sample_id      AFR      EUR  chrom
   0       Sample_1  0.85383  0.14617  chr20
   1       Sample_2  0.68933  0.31067  chr20
   2       Sample_3  1.00000  0.00000  chr20
   3       Sample_4  0.86754  0.13246  chr20
   4       Sample_5  0.68280  0.31720  chr20

You can recover the list of samples or populations exactly as before.

.. code:: python

   sample_ids = g_anc.sample_id.unique().to_arrow()
   pops = g_anc.drop(["sample_id", "chrom"], axis=1).columns.values

``admix``
---------

``admix`` stores the lazy-loaded local ancestry array and mirrors the BED
behavior of ``pandas-plink``.

.. code:: python

   admix

::

   dask.array<concatenate, shape=(646287, 1000), dtype=float32, chunksize=(1024, 256), chunktype=numpy.ndarray>

.. code:: python

   admix.compute()

::

   [[2 2 2 ... 0 0 0]
    [2 2 1 ... 0 0 1]
    [1 2 1 ... 0 0 0]
    ...
    [1 1 2 ... 0 0 0]
    [2 2 2 ... 1 1 1]
    [2 2 1 ... 1 0 1]]

Column ordering follows the population-major convention: all individuals
for the first population, then all individuals for the second, and so on.
This can be reproduced with list comprehensions when building column
labels.

.. code:: python

   col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]

Loci imputation
===============

The imputation utilities now reside in ``rfmix_reader.processing.imputation``
and are exposed at the top level as ``interpolate_array``. The function fills
missing local ancestry loci on an arbitrary variant grid and writes a Zarr store
(``<zarr_outdir>/local-ancestry.zarr``) with shape ``(variants, samples,
ancestries)``.

**Inputs**

* ``variant_loci_df``: pandas DataFrame describing the target variant grid.
  Include ``chrom``/``pos`` and an ``i`` column marking the source row in the
  RFMix output. Any row with ``i`` set to ``NaN`` is interpreted as a missing
  locus that should be interpolated. Sort by genomic coordinate, and ensure a
  ``pos`` column is present when using base-pair interpolation.
* ``admix``: local ancestry array returned by :func:`read_rfmix` with shape
  ``(loci, samples, ancestries)``.
* ``zarr_outdir``: directory where ``local-ancestry.zarr`` will be created.

**Key options**

* ``interpolation`` can be ``"linear"`` (default), ``"nearest"``, or
  ``"stepwise"``.
* ``use_bp_positions=True`` interpolates along ``variant_loci_df['pos']`` rather
  than treating loci as evenly spaced indices.
* ``chunk_size`` and ``batch_size`` control how many rows are materialized per
  interpolation or write step to balance speed and memory use.

**Workflow example**

.. code:: python

   from pathlib import Path
   import pandas as pd
   from rfmix_reader import interpolate_array, read_rfmix

   # Local ancestry loci and trajectories
   loci_df, _, admix = read_rfmix("two_pops/out/", binary_dir="./binary_files")

   # Variant grid: provide chrom/pos plus the RFMix row index in column ``i``
   variant_df = pd.read_parquet("genotypes/variants.parquet")
   variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"]).sort_values("pos")
   variant_loci_df = (
       variant_df.merge(loci_df.to_pandas(), on=["chrom", "pos"], how="outer", indicator=True)
                  .loc[:, ["chrom", "pos", "i", "_merge"]]
   )

   z = interpolate_array(
       variant_loci_df,
       admix,
       zarr_outdir=Path("./imputed_local_ancestry"),
       interpolation="nearest",
       use_bp_positions=True,
       chunk_size=50_000,
   )

``interpolate_array`` automatically uses CUDA (via ``cupy``) when available and
falls back to NumPy otherwise. Interpolation operates on diploid-summed
trajectories and preserves the ancestry dimension.

Visualization
=============

``rfmix-reader`` bundles Matplotlib/Seaborn helpers so that common plots
and exports do not require manual reshaping.

Global summaries
----------------

``plot_global_ancestry`` builds stacked bar plots of the global ancestry
means per individual, while ``plot_ancestry_by_chromosome`` shows
chromosome-level distributions as boxplots.

.. code:: python

   from rfmix_reader import (
       plot_global_ancestry,
       plot_ancestry_by_chromosome,
   )

   plot_global_ancestry(g_anc, dpi=300, bbox_inches="tight")
   plot_ancestry_by_chromosome(g_anc, dpi=300, bbox_inches="tight")

Both accept ``save_path`` and ``save_multi_format`` arguments so you can
export simultaneously to PNG/PDF or rely on interactive rendering by
passing ``save_path=None``.

Export for TAGORE and BED-based tools
-------------------------------------

``generate_tagore_bed`` converts the trio of ``(loci_df, g_anc, admix)``
into a BED table annotated with colors and chromosome copy information,
matching TAGORE's expected columns. ``save_multi_format`` remains
available if you build additional custom Matplotlib figures.

.. code:: python

   from rfmix_reader import generate_tagore_bed

   tagore_df = generate_tagore_bed(loci_df, g_anc, admix, sample_num=0)
   tagore_df.head()
