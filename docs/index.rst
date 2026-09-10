.. rfmix-reader documentation master file, created by
   sphinx-quickstart on Sun Jun  2 10:18:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to rfmix-reader's documentation!
========================================

`rfmix-reader` gives fast, memory-light access to local-ancestry output —
RFMix (``.msp.tsv``, ``.fb.tsv``), FLARE and haptools simulations — as one
lazily-evaluated :class:`xarray.Dataset` backed by a per-chromosome Zarr cache:
one streaming pass over the source, then instant reopen, with position
queries, BED intervals, Parquet export, interpolation and gnomix-style
phasing on the same object.

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   install
   usage
   api

********
Citation
********

If using, please cite the following pre-print:

Kynon JM Benjamin. "RFMix-reader: Accelerated reading and processing for
local ancestry studies." *bioRxiv*. 2024.
DOI: `10.1101/2024.07.13.603370 <https://www.biorxiv.org/content/10.1101/2024.07.13.603370v2>`_

*****************
Comments and bugs
*****************

You can get the source code and open issues `on Github.`_

.. _on Github.: https://github.com/heart-gen/rfmix_reader

Indices and tables
==================
* :ref:`genindex`
* :ref:`search`
