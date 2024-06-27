rfmix\_reader.export\_loci\_admix\_to\_bed
==========================================

.. function:: export_loci_admix_to_bed(loci: DataFrame, rf_q: DataFrame, admix: Array, output_dir: str = "bed_results", verbose: bool = True) -> None

    Export loci and admixture data to a BED (Browser Extensible Data) file.

    This function processes genetic loci data along with admixture proportions and writes the results to a BED format file. The output file includes sample-specific admixture proportions for each population.

    :param loci: A DataFrame containing genetic loci information. Expected to have columns for chromosome, position, and other relevant genetic markers.
    :type loci: pandas.DataFrame
    :param rf_q: A DataFrame containing sample and population information. Used to derive sample IDs and population names.
    :type rf_q: pandas.DataFrame
    :param admix: A Dask Array containing admixture proportions. The shape should be compatible with the number of loci and populations.
    :type admix: dask.array.Array
    :param output_dir: The path for the output BED file. Default is "bed_results".
    :type output_dir: str, optional
    :param verbose: If True, show progress information; if False, suppress progress information. Default is True.
    :type verbose: bool, optional

    :returns: None

    Notes
    -----
    - The function internally calls _generate_bed() to perform the actual file writing.
    - Column names in the output file are formatted as "{sample}_{population}".
    - The output file includes data for all chromosomes present in the input loci DataFrame.
    - Large datasets may require significant processing time and disk space.

    Example
    -------
    ::

        loci, rf_q, admix = read_rfmix(prefix_path)
        export_loci_admix_to_bed(loci_df, rf_q_df, admix_array)

