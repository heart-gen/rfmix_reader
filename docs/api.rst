***
API
***

.. currentmodule:: rfmix_reader

Core I/O
--------

.. autosummary::
   :toctree: api/

   create_binaries
   read_rfmix
   read_flare
   read_simu
   write_data

Utility helpers
---------------

.. autosummary::
   :toctree: api/

   Chunk
   get_pops
   get_prefixes
   get_sample_names
   set_gpu_environment
   delete_files_or_directories
   read_fb

Ancestry data handling
----------------------

.. autosummary::
   :toctree: api/

   admix_to_bed_individual
   generate_tagore_bed

Imputation
----------

.. autosummary::
   :toctree: api/

   interpolate_array
   _expand_array

Visualization
-------------

.. autosummary::
   :toctree: api/

   plot_global_ancestry
   plot_ancestry_by_chromosome
   plot_local_ancestry_tagore
   save_multi_format

Exceptions
----------

.. autosummary::
   :toctree: api/

   BinaryFileNotFoundError

Constants
---------

.. autosummary::
   :toctree: api/
   :nosignatures:

   CHROM_SIZES
   COORDINATES
