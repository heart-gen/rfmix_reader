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

   # pass the store + annotations into read_rfmix phasing
   read_rfmix(
       "../examples/two_populations/out/",
       phase=True,
       phase_ref_zarr_root="refs",
       phase_sample_annot_path="sample_annotations.tsv",
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

To phase-correct local ancestry before stacking across chromosomes, pass
``phase=True`` along with the required reference inputs:

.. code:: python

   loci_df, g_anc, admix = read_rfmix(
       "../examples/two_populations/out/",
       phase=True,
       phase_vcf_path="/path/to/reference.vcf.gz",
       phase_sample_annot_path="/path/to/sample_annot.tsv",
   )

Step-by-step phasing tutorial
-----------------------------

The phasing helpers wrap the RFMix VCF and sample annotations so you can
align haplotypes across chromosomes before downstream analysis. Follow
the checklist below when working with RFMix outputs. FLARE outputs are
already phased and should **skip** this entire section.

1. **Prepare phasing inputs**

   - Confirm that each RFMix ``*.fb.tsv`` file has a matching binary copy
     in ``binary_dir``. If not, generate them with
     ``rfmix_reader.create_binaries`` or the ``create-binaries`` CLI
     shown above.
   - Collect the phased reference VCF that matches the cohort. The path
     to this file is passed via ``phase_vcf_path``.
   - Provide the sample annotation TSV expected by RFMix, typically
     containing ``ID`` and population columns. Point
     ``phase_sample_annot_path`` to this file.

2. **Invoke phasing**

   - **Python API:** set ``phase=True`` when calling ``read_rfmix`` and
     pass both file paths.

     .. code:: python

        loci_df, g_anc, admix = read_rfmix(
            "../examples/two_populations/out/",
            binary_dir="../examples/two_populations/out/binary_files",
            phase=True,
            phase_vcf_path="/path/to/reference.vcf.gz",
            phase_sample_annot_path="/path/to/sample_annot.tsv",
        )

   - **Command-line:** use the CLI entry point to phase during binary
     creation. The arguments mirror the Python call.

     .. code:: shell

        create-binaries \
            --binary_dir ../examples/two_populations/out/binary_files \
            --phase \
            --phase_vcf_path /path/to/reference.vcf.gz \
            --phase_sample_annot_path /path/to/sample_annot.tsv \
            ../examples/two_populations/out/

3. **Interpret the phased output**

   - ``loci_df`` is unchanged aside from any chromosome renaming you
     might perform.
   - ``g_anc`` remains the global ancestry table but now aligns with the
     phased haplotypes produced per chromosome.
   - ``admix`` stores phase-corrected local ancestry calls. The
     population-major column ordering is preserved, so downstream code
     that expects ``(sample, population)`` pairing continues to work.

4. **Troubleshooting tips**

   - If the phased VCF lacks contig lengths or sample names differ from
     the RFMix annotations, fix those before retrying. Alignment is
     strict because the phasing step matches haplotype labels between the
     reference VCF and the RFMix outputs.

``g_anc`` is the canonical variable name returned by ``read_rfmix`` and
is used throughout the visualization helpers described later.

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

Imputing local ancestry loci information to genotype variant locations
improves integration of the local ancestry information with genotype
data. As such, we also provide the ``interpolate_array`` function to
efficiently interpolate missing values when local ancestry loci
information is converted to more variable genotype variant locations. It
leverages the power of
`Zarr <https://zarr.readthedocs.io/en/stable/index.html>`_ arrays, making
it suitable for handling substantial datasets while managing memory usage
effectively.

**Note**: Following imputation, ``variant_df`` will include genomic
positions for both local ancestry and genotype data.

.. code:: python

   def _load_genotypes(plink_prefix_path):
       from tensorqtl import pgen
       pgr = pgen.PgenReader(plink_prefix_path)
       variant_df = pgr.variant_df
       variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
       return pgr.load_genotypes(), variant_df

   def _load_admix(prefix_path, binary_dir):
       from rfmix_reader import read_rfmix
       return read_rfmix(prefix_path, binary_dir=binary_dir)

.. code:: python

   from rfmix_reader import interpolate_array
   basename = "/projects/b1213/large_projects/brain_coloc_app/input"
   # Local ancestry
   prefix_path = f"{basename}/local_ancestry_rfmix/_m/"
   binary_dir = f"{basename}/local_ancestry_rfmix/_m/binary_files/"
   loci, _, admix = _load_admix(prefix_path, binary_dir)
   loci.rename(columns={"chromosome": "chrom",
                        "physical_position": "pos"},
               inplace=True)
   # Variant data
   plink_prefix = f"{basename}/genotypes/TOPMed_LIBD"
   _, variant_df = _load_genotypes(plink_prefix)
   variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
                                           keep='first')
   # Keep all locations for more accurate imputation
   variant_loci_df = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
                                      how="outer", indicator=True)\
                               .loc[:, ["chrom", "pos", "i", "_merge"]]
   data_path = f"{basename}/local_ancestry_rfmix/_m"
   z = interpolate_array(variant_loci_df, admix, data_path)

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
