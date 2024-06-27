rfmix\_reader.Chunk
===================

.. class:: Chunk(nsamples: Optional[int] = 1024, nloci: Optional[int] = 1024)

    Chunk specification for a contiguous submatrix of the haplotype matrix.

    :param nsamples: Number of samples in a single chunk, limited by the total number of samples. Set to `None` to include all samples. Default is 1024.
    :type nsamples: Optional[int]
    :param nloci: Number of loci in a single chunk, limited by the total number of loci. Set to `None` to include all loci. Default is 1024.
    :type nloci: Optional[int]

    Notes
    -----
    - Small chunks may increase computational time, while large chunks may increase memory usage.
    - For small datasets, try setting both `nsamples` and `nloci` to `None`.
    - For large datasets where you need to use every sample, try setting `nsamples=None` and choose a small value for `nloci`.
