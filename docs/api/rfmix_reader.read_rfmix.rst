rfmix\_reader.read\_rfmix
=========================

.. function:: read_rfmix(file_prefix: str, binary_dir: str = "./binary_files", generate_binary: bool = False, verbose: bool = True) -> Tuple[pandas.DataFrame, pandas.DataFrame, dask.array.Array]

    Read RFMix files into data frames and a Dask array.

    :param file_prefix: Path prefix to the set of RFMix files. It will load all of the chromosomes at once.
    :type file_prefix: str
    :param binary_dir: Path prefix to the binary version of RFMix (*fb.tsv) files. Default is "./binary_files".
    :type binary_dir: str, optional
    :param generate_binary: If True, generate the binary file. Default is False.
    :type generate_binary: bool, optional
    :param verbose: If True, show progress information; if False, suppress progress information. Default is True.
    :type verbose: bool, optional

    :returns:
        - **loci** (*pandas.DataFrame*) - Loci information for the FB data.
        - **rf_q** (*pandas.DataFrame*) - Global ancestry by chromosome from RFMix.
        - **admix** (*dask.array.Array*) - Local ancestry per population (columns pop1*nsamples ... popX*nsamples). This is in order of the populations see `rf_q`.

    :rtype: Tuple[pandas.DataFrame, pandas.DataFrame, dask.array.Array]

    **Notes**

    Local ancestry can be either 0, 1, 2, or math.nan:

    - **0**: No alleles are associated with this ancestry
    - **1**: One allele is associated with this ancestry
    - **2**: Both alleles are associated with this ancestry
