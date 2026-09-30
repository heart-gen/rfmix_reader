API Reference
=============

Readers and cache
-----------------

.. autosummary::
   :toctree: api/generated

   rfmix_reader.open_rfmix
   rfmix_reader.open_flare
   rfmix_reader.open_simu
   rfmix_reader.open_local_ancestry
   rfmix_reader.convert

The Dataset
-----------

Every reader returns an :class:`xarray.Dataset` that follows
:mod:`rfmix_reader.core.schema`; ``ds.la`` (:class:`rfmix_reader.core.accessor.LocalAncestryAccessor`)
provides the views and operations.

.. autosummary::
   :toctree: api/generated

   rfmix_reader.core.schema
   rfmix_reader.core.accessor.LocalAncestryAccessor
   rfmix_reader.core.codes
   rfmix_reader.core.zarr_io
   rfmix_reader.core.legacy.from_legacy

Operations
----------

.. autosummary::
   :toctree: api/generated

   rfmix_reader.ops.bed.to_bed
   rfmix_reader.ops.positions.at_positions
   rfmix_reader.ops.positions.locus_index
   rfmix_reader.ops.positions.counts_at
   rfmix_reader.ops.positions.chromosome_index
   rfmix_reader.ops.parquet.to_parquet
   rfmix_reader.ops.interpolate.interpolate
   rfmix_reader.ops.interpolate.build_variant_grid
   rfmix_reader.ops.tagore.to_tagore

Phasing and imputation
----------------------

.. autosummary::
   :toctree: api/generated

   rfmix_reader.processing.phase.PhasingConfig
   rfmix_reader.processing.phase.phase_dataset
   rfmix_reader.processing.phase.phase_haplotypes
   rfmix_reader.processing.phase.gnomix_switch_mask_sample
   rfmix_reader.processing.phase.phase_rfmix_chromosome_to_zarr
   rfmix_reader.processing.phase.merge_phased_zarrs
   rfmix_reader.processing.phase.build_reference_haplotypes_from_zarr
   rfmix_reader.processing.imputation.interpolate_array
   rfmix_reader.processing.constants

Formats
-------

.. autosummary::
   :toctree: api/generated

   rfmix_reader.formats.discover
   rfmix_reader.formats.base
   rfmix_reader.formats.common
   rfmix_reader.formats.rfmix_msp
   rfmix_reader.formats.rfmix_fb
   rfmix_reader.formats.flare_vcf
   rfmix_reader.formats.haptools_vcf
   rfmix_reader.formats.global_ancestry

Visualisation
-------------

.. autosummary::
   :toctree: api/generated

   rfmix_reader.viz.visualization.plot_global_ancestry
   rfmix_reader.viz.visualization.plot_ancestry_by_chromosome
   rfmix_reader.viz.visualization.save_multi_format
   rfmix_reader.viz.tagore.plot_local_ancestry_tagore

Reference panels, backends, command line
----------------------------------------

.. autosummary::
   :toctree: api/generated

   rfmix_reader.io.prepare_reference
   rfmix_reader.backends
   rfmix_reader.cli.main
