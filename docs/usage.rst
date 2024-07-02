.. raw:: org

   #+PROPERTY:  header-args: :dir /dcs04/lieber/statsgen/jbenjami/tutorials/eqtl_analysis_tutorial

.. raw:: org

   #+PROPERTY:  header-args:R :cache yes :exports both :session *R* :eval never-export

.. raw:: org

   #+PROPERTY:  header-args:python :session *Python* :cache yes :exports both :eval never-export

.. raw:: org

   #+PROPERTY:  header-args:sh :cache yes :exports both :eval never-export

.. raw:: org

   #+STARTUP:   align fold nodlcheck hidestars oddeven lognotestate

.. raw:: org

   #+TAGS:      Write(w) Update(u) Fix(f) Check(c) noexport(n)

Haplotypes
==========

Using ``rfmix-reader`` is as simple as
`pandas-plink <https://pandas-plink.readthedocs.io/en/latest/usage.html>`__.

We provide example data for two and three population admixtured from
simulation data created with
`Haptools <https://haptools.readthedocs.io/en/stable/>`__. You can
download it from Figshare:

Once downloaded, we are ready to start!

Input
-----

First, we need to generate binary files. I suggest using
``create_binaries``.

.. code:: python

   from rfmix_reader import create_binaries

   prefix_path = "../examples/two_populations/out/"
   create_binaries(prefix_path)

.. raw:: org

   #+RESULTS[20f284c23771c5bf268a8cb4a468752edf5b0410]:

::

   Created binary files at: ./binary_files
   Converting fb files to binary!
     0% 0/3 [00:00<?, ?it/s] 33% 1/3 [05:03<10:07, 303.82s/it]100% 3/3 [05:03<00:00, 101.27s/it]
   Successfully converted 3 files to binary format.

Once the binary files are created, we can read in the data with the main
function ``read_rfmix``.

.. code:: python

   from rfmix_reader import read_rfmix

   loci, rf_q, admix = read_rfmix(prefix_path)

.. raw:: org

   #+RESULTS[894f008f7dfcb07d33816de2f9c4858756db92f6]:

::

   GPU 0: NVIDIA TITAN V
     Total memory: 11.77 GB
     CUDA capability: 7.0
   Multiple files read in this order: ['chr20', 'chr21', 'chr22']
   Mapping loci files:   0% 0/3 [00:00<?, ?it/s]Mapping loci files:  33% 1/3 [00:02<00:05,  2.72s/it]Mapping loci files:  67% 2/3 [00:04<00:01,  1.93s/it]Mapping loci files: 100% 3/3 [00:05<00:00,  1.73s/it]Mapping loci files: 100% 3/3 [00:05<00:00,  1.86s/it]
   Mapping Q files:   0% 0/3 [00:00<?, ?it/s]Mapping Q files: 100% 3/3 [00:00<00:00, 47.69it/s]
   Mapping fb files:   0% 0/3 [00:00<?, ?it/s]Mapping fb files:  33% 1/3 [00:00<00:00,  2.66it/s]Mapping fb files:  67% 2/3 [00:00<00:00,  3.46it/s]Mapping fb files: 100% 3/3 [00:00<00:00,  3.75it/s]Mapping fb files: 100% 3/3 [00:00<00:00,  3.55it/s]

With a GPU, three chromosomes can be loaded in to your session in less
than a minute.

Output
------

``loci``
~~~~~~~~

``loci`` are the metadata for the RFMix results.

.. code:: python

   loci.shape

.. raw:: org

   #+RESULTS[217b70fa31fcce528d45f44213a25d1722e1309b]:

::

   (646287, 3)

.. code:: python

   loci

.. raw:: org

   #+RESULTS[bc9ff363ba2f5069d7ad629933ab8302c74b7f5c]:

::

          chromosome  physical_position       i
   0           chr20              60137       0
   1           chr20              60291       1
   2           chr20              60340       2
   3           chr20              60440       3
   4           chr20              60823       4
   ...           ...                ...     ...
   646282      chr22           50790690  646282
   646283      chr22           50790993  646283
   646284      chr22           50791163  646284
   646285      chr22           50791228  646285
   646286      chr22           50791360  646286

   [646287 rows x 3 columns]

To model it after ``pandas_plink``, there is an index column ``i``. This
is useful for software developing, but in general only the first two
columns are needed.

``rf_q``
~~~~~~~~

``rf_q`` is the global ancestry results per chromosome for each
individual. This is the ``*.rfmix.Q`` files combined into a single
``DataFrame``.

.. code:: python

   rf_q.shape

.. raw:: org

   #+RESULTS[03374a9f07046dd7deeef0520f12f85217cf8c20]:

::

   (1500, 4)

.. code:: python

   rf_q

.. raw:: org

   #+RESULTS[d52da46fcc3adf7aa1e9dfa5442db27cc50082af]:

::

          sample_id      AFR      EUR  chrom
   0       Sample_1  0.85383  0.14617  chr20
   1       Sample_2  0.68933  0.31067  chr20
   2       Sample_3  1.00000  0.00000  chr20
   3       Sample_4  0.86754  0.13246  chr20
   4       Sample_5  0.68280  0.31720  chr20
   ...          ...      ...      ...    ...
   1495  Sample_496  0.82322  0.17678  chr22
   1496  Sample_497  0.73456  0.26544  chr22
   1497  Sample_498  1.00000  0.00000  chr22
   1498  Sample_499  0.87362  0.12638  chr22
   1499  Sample_500  0.85129  0.14871  chr22

   [1500 rows x 4 columns]

Since we have three chromosomes, that means there are 500 samples in
this example dataset.

.. code:: python

   rf_q.groupby("chrom").size()

.. raw:: org

   #+RESULTS[d92e8d18e5bbb94760735575df8b58cf442f61c1]:

::

   chrom
   chr22    500
   chr20    500
   chr21    500
   dtype: int64

Let's exact the sample names! This is a ``cudf`` DataFrame, so we need
to extract the data with ``.to_arrow()``. When running on CPU, this will
be a regular ``pandas`` DataFrame.

.. code:: python

   type(rf_q)

.. raw:: org

   #+RESULTS[31f076edd1d8a293467b76d46381391573fd01ac]:

::

   <class 'cudf.core.dataframe.DataFrame'>

.. code:: python

   sample_ids = rf_q.sample_id.unique().to_arrow()
   len(sample_ids)

.. raw:: org

   #+RESULTS[d3e5cab41b367de4cab44d2d0450f1b958f4d098]:

::

   500

We'll also get the unique populations.

.. code:: python

   pops = rf_q.drop(["sample_id", "chrom"], axis=1).columns.values
   pops

.. raw:: org

   #+RESULTS[943d0f4206518c373fa852ab000059693e2b2897]:

::

   ['AFR' 'EUR']

``admix``
~~~~~~~~~

``admix`` is the convert RFMix results from the ``*.fb.tsv`` files.
Here, we add the alleles and re-subset the data so that the first
population is first (all samples) followed by the next, and the next.
This means instead of 0 and 1, you can get 0, 1, or 3.

.. code:: python

   admix

.. raw:: org

   #+RESULTS[786d091553720e67cc5780ad7bbd2265492be434]:

::

   dask.array<concatenate, shape=(646287, 1000), dtype=float32, chunksize=(1024, 256), chunktype=numpy.ndarray>

To reduce memory consumption, this large data is held in a dask array,
exactly like ``pandas_plink`` BED data.

.. code:: python

   admix.compute()

.. raw:: org

   #+RESULTS[070fc2065a660e8042230bf7713804fdb124fbba]:

::

   [[2 2 2 ... 0 0 0]
    [2 2 1 ... 0 0 1]
    [1 2 1 ... 0 0 0]
    ...
    [1 1 2 ... 0 0 0]
    [2 2 2 ... 1 1 1]
    [2 2 1 ... 1 0 1]]

.. code:: python

   admix.shape

.. raw:: org

   #+RESULTS[19574afcca5d5cbc89e58eb226076e4ed3afeab7]:

::

   (646287, 1000)

The rows are the same as the ``loci`` data, in the sample order.

.. code:: python

   loci.shape

.. raw:: org

   #+RESULTS[217b70fa31fcce528d45f44213a25d1722e1309b]:

::

   (646287, 3)

The rows are the total samples x number of populations. This is in a
specific order. All samples are grouped by population instead of by the
sample.

.. code:: python

   col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
   len(col_names)

.. raw:: org

   #+RESULTS[6d3b0a823d116490484f2500f47ebbb03fcd208c]:

::

   1000

.. code:: python

   col_names[0:4]

.. raw:: org

   #+RESULTS[c8ca5d8c680865988858e9cafb571adceb27970d]:

::

   ['Sample_1_AFR', 'Sample_2_AFR', 'Sample_3_AFR', 'Sample_4_AFR']

.. code:: python

   col_names[500:504]

.. raw:: org

   #+RESULTS[9889ae17959e0911178a53e41e70d58d7ce11224]:

::

   ['Sample_1_EUR', 'Sample_2_EUR', 'Sample_3_EUR', 'Sample_4_EUR']

This is the correct order for the admix array data.
