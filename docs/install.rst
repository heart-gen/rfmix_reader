*******
Install
*******

::

  pip install rfmix-reader

Python 3.11 or newer.  The core depends on ``numpy``, ``pandas``, ``dask``,
``xarray``, ``zarr`` and ``cyvcf2`` only.

Extras
------

============= ================================================ ==========================================
Extra         Adds                                             Needed for
============= ================================================ ==========================================
``viz``       matplotlib, seaborn, cairosvg                    ``plot_*`` functions, ``ds.la.to_tagore``
``io``        pyarrow                                          ``ds.la.to_parquet``
``reference`` bio2zarr                                         ``prepare-reference`` (VCF to VCF-Zarr)
``gpu``       torch, cupy-cuda12x, cudf-cu12, dask-cudf-cu12   optional CuPy compute backend
``all``       everything above
============= ================================================ ==========================================

::

  pip install "rfmix-reader[viz,io]"

GPU acceleration
----------------

Parsing is CPU-only by design (it is I/O bound).  When CuPy is importable the
imputation and plotting helpers use it as the array backend; see
:mod:`rfmix_reader.backends`.  The ``gpu`` extra targets CUDA 12 wheels — pick
the matching build strings for other CUDA versions.

Development
-----------

::

  git clone https://github.com/heart-gen/rfmix_reader
  cd rfmix_reader
  poetry install --with test --extras "viz io"
  poetry run pytest             # fast suite (seconds)
  poetry run pytest --run-slow  # also the chr21 tests (needs git-LFS data)
