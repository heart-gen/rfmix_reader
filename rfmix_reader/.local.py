# This script process local ancestry data generated from
# RFMix v2.
# Documentation improved with LLM AI
__author__ = "Kynon J Benjamin"

import cudf as pd
import numpy as np
from rfmix_reader import read_rfmix, interpolate_array

################### testing #############################

def get_test_ids():
    test_ids = "../example/sample_id_to_brnum.tsv"
    return pd.read_csv(test_ids, sep="\t", usecols=[1])


def test_data():
    select_samples = list(get_test_ids().BrNum.to_pandas())
    prefix_path = "/projects/p32505/projects/localQTL-software/" +\
                  "local_ancestry_rfmix/_m/"
    binary_dir = f"{prefix_path}/binary_files"
    loci, rf_q, admix = read_rfmix(
        prefix_path, verbose=True, binary_dir=binary_dir)
    rfr = RFMixReader(prefix_path, select_samples=select_samples)

########################################################


class RFMixReader(object):
    """
    Class for reading loci from RFMix files.

    Parameters
    ----------
    prefix_path : str
        The prefix path to the RFMix files (e.g., fb.tsv, rfmix.Q).
    select_samples : list of str, optional
        A list of sample IDs to select a subset of samples.
        If None, all samples are included.
    exclude_chrs : list of str, optional
        A list of chromosome names to exclude from the data.
        If None, no chromosomes are excluded.
    binary_dir_name : str, optional
        The directory name where the binary files are stored.
        Defaults to "binary_files".
    verbose : bool, optional
        Whether to print verbose messages during processing.
        Defaults to True.
    dtype : numpy.dtype, optional
        The data type to use for numerical arrays. Defaults to np.int8.

    Notes
    -----
    Use the following command to generate RFMix files per chromosome:
        rfmix -f {rfmix_prefix_path}.vcf.gz \
              -r {reference.vcf.gz} \
              -m {samples_ids} \
              -g {genetic_map} \
              -o {rfmix_prefix_path}.chr$i \
              --chromosome=chr$i

    Use the following command to convert and generate binary files
    on the command line:
        create-binaries./ --binary_dir "binary_files"

    In Python:
        create_binaries(file_path, binary_dir)

    This class uses the `read_rfmix` function from `rfmix_reader`.
    """

    def __init__(self, prefix_path, select_samples=None, exclude_chrs=None,
                 binary_dir_name="binary_files", verbose=True, dtype=np.int8):
        """
        Initialize the RFMixReader object.

        Parameters
        ----------
        prefix_path : str
            The prefix path to the RFMix files.
        select_samples : list of str, optional
            A list of sample IDs to select a subset of samples.
        exclude_chrs : list of str, optional
            A list of chromosome names to exclude from the data.
        binary_dir_name : str, optional
            The directory name where the binary files are stored.
        verbose : bool, optional
            Whether to print verbose messages during processing.
        dtype : numpy.dtype, optional
            The data type to use for numerical arrays.

        Returns
        -------
        None
        """
        self.bin_dir = f"{prefix_path}/{binary_dir_name}"
        self.loci, self.rf_q, self.admix = read_rfmix(prefix_path,
                                                      binary_dir=self.bin_dir,
                                                      verbose=verbose)
        # Create a haplotype ID column by concatenating chromosome and
        # physical position
        self.loci["hap"] = self.loci['chromosome'].astype(
            str) + '_' + self.loci['physical_position'].astype(str)
        # Select samples from chromosome 1 for gsam
        self.gsam = self.rf_q[(self.rf_q["chrom"] == "chr1")].copy()
        self.n_loci = self.loci.shape[0]
        # Get sample IDs
        self.sample_ids = self._get_sample_ids(self.gsam)
        self.n_pops = self._get_n_pop()
        # Select a subset of samples if specified
        if select_samples is not None:
            ix = [self.sample_ids.index(i) for i in select_samples]
            self.gsam = self.gsam.loc[ix].copy()
            self.admix = self.admix[:, ix]
            self.sample_ids = self._get_sample_ids(self.gsam)
        # Exclude specified chromosomes
        if exclude_chrs is not None:
            mask = ~self.loci['chromosome'].isin(exclude_chrs)
            self.admix = self.admix[mask.values, :]
            self.loci = self.loci[mask].copy()
            self.loci.reset_index(drop=True, inplace=True)
            self.loci['i'] = self.loci.index
        # Currently works for 2 populations only
        self.n_samples = self.gsam.shape[0]  # * self.n_pop
        # Get unique chromosomes
        self.chrs = self._get_chroms(self.loci)
        # Create haplotype dataframe
        hap_df = self.loci.set_index('hap')[['chromosome', 'physical_position']]
        hap_df['index'] = np.arange(hap_df.shape[0])
        self.hap_df = hap_df
        self.hap_dfs = {c:g[['physical_position', 'index']] for c,g in hap_df.groupby('chromosome', sort=False)}

    def _get_sample_ids(self, df):
        """
        Get sample IDs from a DataFrame.

        Parameters
        ----------
        df : cudf.DataFrame or pd.DataFrame
            The DataFrame containing sample IDs.

        Returns
        -------
        list of str
            A list of sample IDs.
        """
        if isinstance(df, (pd.DataFrame, pd.Series)):
            return df["sample_id"].to_arrow().tolist()
        else:
            return df["sample_id"].tolist()

    def _get_chroms(self, loci):
        """
        Get unique chromosomes from the loci DataFrame.

        Parameters
        ----------
        loci : cudf.DataFrame or pandas.DataFrame
            The loci DataFrame.

        Returns
        -------
        list of str
            A list of unique chromosome names.
        """
        if isinstance(loci, (pd.DataFrame, pd.Series)):
            return loci["chromosome"].to_pandas().tolist()
        else:
            return loci['chromosome'].unique().tolist()

    def _get_n_pop(self):
        """
        Calculate and return the number of populations.

        Returns:
        --------
        int
            The number of populations in the data.

        Notes:
        ------
        This method assumes that the number of columns in self.admix
        is a multiple of the number of samples. It divides the number
        of columns in self.admix by the number of samples to get the
        number of populations.
        """
        n_samples = len(self.sample_ids)
        if n_samples == 0:
            raise ValueError("No samples found in the data.")
        n_columns = self.admix.shape[1]
        if n_columns % n_samples != 0:
            raise ValueError(
                "The number of columns in admix data is not a multiple of the number of samples.")
        return n_columns // n_samples

    def get_region_index(self, region_str, return_pos=False):
        """
        Get the indices for a specified genomic region.

        Parameters
        ----------
        region_str : str
            A string specifying the genomic region (e.g.,
            'chr1:1000-2000' or 'chr1').
        return_pos : bool, optional
            Whether to return physical positions along with indices.
            Defaults to False.

        Returns
        -------
        indices : numpy.ndarray or dask.array.core.Array
            The indices corresponding to the specified genomic region.
        pos_s : pandas.Series or cudf.Series, optional
            The physical positions corresponding to the specified
            genomic region if return_pos is True.
        """
        s = region_str.split(':')
        chrom = s[0]
        c = self.loci[self.loci['chromosome'] == chrom]
        if len(s) > 1:
            start, end = map(int, s[1].split('-'))
            c = c[(c['physical_position'] >= start)
                  & (c['physical_position'] <= end)]
        indices = c.index.values
        if return_pos:
            return indices, c.set_index('hap')['physical_position']
        else:
            return indices

    def get_region(self, region_str, sample_ids=None, verbose=False,
                   dtype=np.int8):
        """
        Get the genotype data for a specified genomic region.

        Parameters
        ----------
        region_str : str
            A string specifying the genomic region. It can be in the
            format 'chrX' or 'chrX:start-end'. For example, 'chr1' or
            'chr1:1000-2000'.
        sample_ids : list of str, optional
            A list of sample IDs to include in the result.
            If None, all samples are included.
    verbose : bool, optional
            Whether to print verbose messages during processing.
            Defaults to False.
        dtype : numpy.dtype, optional
            The data type to use for the returned genotype array.
            Defaults to np.int8.

        Returns
        -------
        g : numpy.ndarray
            The genotype data for the specified genomic region.
        pos_s : pandas.Series or cudf.Series
            The physical positions corresponding to the loci in the
            specified genomic region.

        Notes
        -----
        This method uses the `get_region_index` method to obtain the
        indices of the loci within the specified region and then
        extracts the corresponding genotype data from `self.admix`.

        Examples
        --------
        >>> reader = RFMixReader('path/to/prefix')
        >>> genotypes, positions = reader.get_region('chr1:1000-2000')
        >>> print(genotypes.shape)
        >>> print(positions.head())
        """
        ix, pos_s = self.get_region_index(region_str, return_pos=True)
        g = self.admix[ix].compute().astype(dtype)
        if sample_ids is not None:
            sample_indices = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, sample_indices]
        return g, pos_s

    def get_loci(self, haplotype_ids, sample_ids=None, verbose=False,
                 dtype=np.int8):
        """
        Get the genotype data for a list of specified haplotype IDs.

        Parameters
        ----------
        haplotype_ids : list of str
            A list of haplotype IDs to retrieve genotype data for.
        sample_ids : list of str, optional
            A list of sample IDs to include in the result.
            If None, all samples are included.
        verbose : bool, optional
            Whether to print verbose messages during processing.
            Defaults to False.
        dtype : numpy.dtype, optional
            The data type to use for the returned genotype array.
            Defaults to np.int8.

        Returns
        -------
        g : numpy.ndarray
            The genotype data for the specified haplotype IDs.
        pos_s : pandas.Series or cudf.Series
            The physical positions corresponding to the specified
            haplotype IDs.

        Notes
        -----
        This method filters the loci based on the provided haplotype IDs
        and extracts the corresponding genotype data from `self.admix`.

        Examples
        --------
        >>> reader = RFMixReader('path/to/prefix')
        >>> genotypes, positions = reader.get_loci(['chr1_1000', 'chr1_2000'])
        >>> print(genotypes.shape)
        >>> print(positions.head())
        """
        c = self.loci[self.loci['hap'].isin(haplotype_ids)]
        indices = c.index.values
        g = self.admix[indices].compute().astype(dtype)
        if sample_ids is not None:
            sample_indices = [self.sample_ids.index(i) for i in sample_ids]
            g = g[:, sample_indices]
        return g, c.set_index('hap')['physical_position']

    def get_locus(self, haplotype_id, sample_ids=None, verbose=False,
                  dtype=np.int8):
        """
        Get the genotype data for a single specified haplotype ID.

        Parameters
        ----------
        haplotype_id : str
            The haplotype ID to retrieve genotype data for.
        sample_ids : list of str, optional
            A list of sample IDs to include in the result.
            If None, all samples are included.
        verbose : bool, optional
            Whether to print verbose messages during processing.
            Defaults to False.
        dtype : numpy.dtype, optional
            The data type to use for the returned genotype array.
            Defaults to np.int8.

        Returns
        -------
        g : pandas.Series or cudf.Series
            The genotype data for the specified haplotype ID as a
            pandas Series or cudf.Series.

        Notes
        -----
        This method filters the loci based on the provided haplotype ID
        and extracts the corresponding genotype data from `self.admix`.
        It returns a pandas or cuDF Series with sample IDs as the index.

        Examples
        --------
        >>> reader = RFMixReader('path/to/prefix')
        >>> genotypes = reader.get_locus('chr1_1000')
        >>> print(genotypes.head())
        """
        g, _ = self.get_loci([haplotype_id], sample_ids=sample_ids,
                             verbose=verbose, dtype=dtype)
        if sample_ids is None:
            return pd.Series(g[0], index=self.sample_ids, name=haplotype_id)
        else:
            return pd.Series(g[0], index=sample_ids, name=haplotype_id)

    def load_loci(self):
        """
        Load all loci into memory as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
        A pandas or cudf DataFrame containing all genotype data with
        haplotype IDs as the index and sample IDs as the columns.

        Notes
        -----
        This method loads the entire genotype data into memory, which
        can be memory-intensive for large datasets.
        It is recommended to use this method only when necessary and
        when sufficient memory is available.

        Examples
        --------
        >>> reader = RFMixReader('path/to/prefix')
        >>> genotypes_df = reader.load_loci()
        >>> print(genotypes_df.head())
        """
        return pd.DataFrame(self.admix.compute(), index=self.loci['hap'],
                            columns=self.sample_ids)


def load_loci(prefix_path, select_samples=None,
              binary_dir_name="binary_files"):
    """
    Load loci data from RFMix files and return the genotype data
    along with haplotype information.

    Parameters
    ----------
    prefix_path : str
        The prefix path to the RFMix files (e.g., fb.tsv, rfmix.Q).
    select_samples : list of str, optional
        A list of sample IDs to select a subset of samples.
        If None, all samples are included.
    binary_dir_name : str, optional
        The directory name where the binary files are stored.
        Defaults to "binary_files".

    Returns
    -------
    loci_df : pandas.DataFrame
        A pandas DataFrame containing all genotype data with
        haplotype IDs as the index and sample IDs as the columns.
    hap_df : pandas.DataFrame
        A pandas DataFrame containing haplotype information with
        'hap' as the index and columns 'chromosome' and 'physical_position'.

    Notes
    -----
    This function uses the `RFMixReader` class to read RFMix
    files and load the genotype data into memory.
    It also extracts haplotype information from the `RFMixReader` object.

    Examples
    --------
    >>> loci_df, hap_df = load_loci('path/to/prefix', select_samples=['sample1', 'sample2'])
    >>> print(loci_df.head())
    >>> print(hap_df.head())
    """
    rfr = RFMixReader(prefix_path, select_samples=select_samples,
                      binary_dir_name=binary_dir_name)
    hap_df = rfr.loci.set_index('hap')[['chromosome', 'physical_position']]
    loci_df = rfr.load_loci()
    return loci_df, hap_df


def get_cis_ranges(phenotype_pos_df, chr_variant_dfs,
                   chr_haplotype_dfs, window, verbose=True):
    """
    Calculate genotype and haplotype ranges within the cis-window.

    Parameters:
        phenotype_pos_df: DataFrame defining position of each phenotype.
        chr_variant_dfs: Dictionary of DataFrames mapping variant positions to
                         indices, grouped by chromosome.
        chr_haplotype_dfs: Dictionary of DataFrames mapping haplotype positions
                           to indices, grouped by chromosome.
        window: The size of the cis-window.
        verbose: Whether to print progress.

    Returns:
        cis_ranges: Dictionary of genotype index ranges for each phenotype.
        drop_ids: List of phenotype IDs without variants or haplotypes in the cis-window.
    """
    # check phenotypes & calculate genotype and loci ranges
    if 'pos' in phenotype_pos_df:
        phenotype_pos_df = phenotype_pos_df.rename(columns={'pos':'start'})
        phenotype_pos_df['end'] = phenotype_pos_df['start']
    phenotype_pos_dict = phenotype_pos_df.to_dict(orient='index')

    drop_ids = []
    cis_ranges = {}
    n = len(phenotype_pos_df)
    for k, phenotype_id in enumerate(phenotype_pos_df.index, 1):
        if verbose and (k % 1000 == 0 or k == n):
            print(f'\r  * checking phenotypes: {k}/{n}',  end='' if k != n else None)

        pos = phenotype_pos_dict[phenotype_id]
        chrom = pos['chr']

        # Check for variants within cis-window
        if chrom in chr_variant_dfs:
            variant_m = len(chr_variant_dfs[chrom]['pos'].values)
            variant_lb = bisect.bisect_left(chr_variant_dfs[chrom]['pos'].values, pos['start'] - window)
            variant_ub = bisect.bisect_right(chr_variant_dfs[chrom]['pos'].values, pos['end'] + window)
            if variant_lb != variant_ub:
                variant_r = chr_variant_dfs[chrom]['index']\
                    .values[[variant_lb, variant_ub - 1]]
            else:
                variant_r = []
        else:
            variant_r = []

        # Check for haplotypes within cis-window
        if chrom in chr_haplotype_dfs:
            haplotype_m = len(chr_haplotype_dfs[chrom]['physical_position'].values)
            haplotype_lb = bisect.bisect_left(chr_haplotype_dfs[chrom]['physical_position'].values, pos['start'] - window)
            haplotype_ub = bisect.bisect_right(chr_haplotype_dfs[chrom]['physical_position'].values, pos['end'] + window)
            if haplotype_lb != haplotype_ub:
                haplotype_r = chr_haplotype_dfs[chrom]['index']\
                    .values[[haplotype_lb, haplotype_ub - 1]]
            else:
                haplotype_r = []
        else:
            haplotype_r = []

        # Check if both variants and haplotypes exists
        if len(variant_r) > 0 and len(haplotype_r) > 0:
            cis_ranges[phenotype_id] = {
                "variants": variant_r,
                "haplotypes": haplotype_r
            }
        else:
            drop_ids.append(phenotype_id)

    return cis_ranges, drop_ids


class InputGeneratorCis(object):
    """
    Input generator for cis-mapping

    Inputs:
      genotype_df:      genotype DataFrame (genotypes x samples)
      variant_df:       DataFrame mapping variant_id (index) to chrom, pos
      loci_df:          admixture loci DataFrame (loci x samples)
      hap_df:           DataFrame mapping hap_id (index) to chrom, pos
      phenotype_df:     phenotype DataFrame (phenotypes x samples)
      phenotype_pos_df: DataFrame defining position of each phenotype, with columns ['chr', 'pos'] or ['chr', 'start', 'end']
      window:           cis-window; selects variants within +- cis-window from 'pos' (e.g., TSS for gene-based features)
                        or within [start-window, end+window] if 'start' and 'end' are present in phenotype_pos_df

    Generates: phenotype array, genotype array (2D), cis-window indices, phenotype ID
    """
    def __init__(self, genotype_df, variant_df, phenotype_df, phenotype_pos_df,
                 loci_df, hap_df, group_s=None, window=1000000):
        self.genotype_df = genotype_df
        self.variant_df = variant_df.copy()
        self.variant_df['index'] = np.arange(variant_df.shape[0])
        self.loci_df = loci_df
        self.hap_df = hap_df.copy()
        self.hap_df["index"] = np.arange(hap_df.shape[0])
        self.n_samples = phenotype_df.shape[1]
        self.group_s = group_s
        self.window = window

        self._validate_data()
        self._filter_phenotypes()
        self._calculate_cis_ranges()

    def _validate_data(self):
        assert (genotype_df.index == variant_df.index).all(), "Genotype and variant DataFrames must have the same index."
        assert (phenotype_df.index == phenotype_df.index.unique()).all(), "Phenotype DataFrame index must be unique."
        assert (loci_df.index == hap_df.index).all(), "Loci and haplotype DataFrames must have the same index."

    def _filter_phenotypes(self):
        # Separate filtering for genotypes and haplotypes
        self._filter_by_genotypes()
        self._filter_by_haplotypes()

        # check for constant phenotypes and drop
        m = np.all(self.phenotype_df.values == self.phenotype_df.values[:,[0]], 1)
        if m.any():
            print(f'    ** dropping {np.sum(m)} constant phenotypes')
            self.phenotype_df = self.phenotype_df.loc[~m]
            self.phenotype_pos_df = self.phenotype_pos_df.loc[~m]

        if len(self.phenotype_df) == 0:
            raise ValueError("No phenotypes remain after filters.")

    def _filter_by_genotypes(self):
        # drop phenotypes without genotypes on same contig
        variant_chrs = self.variant_df['chrom'].unique()
        phenotype_chrs = self.phenotype_pos_df['chr'].unique()
        self.chrs = [i for i in phenotype_chrs if i in variant_chrs]
        m = phenotype_pos_df['chr'].isin(self.chrs)
        if any(~m):
            print(f'    ** dropping {sum(~m)} phenotypes on chrs. without genotypes')
        self.phenotype_df = phenotype_df[m]
        self.phenotype_pos_df = phenotype_pos_df[m]

    def _filter_by_haplotypes(self):
        # drop phenotypes without haplotypes on same contig
        hap_chrs = self.hap_df['chromosomes'].unique()
        phenotype_chrs = self.phenotype_pos_df['chr'].unique()
        self.chrs = [i for i in phenotype_chrs if i in hap_chrs]
        m = phenotype_pos_df['chr'].isin(self.chrs)
        if any(~m):
            print(f'    ** dropping {sum(~m)} phenotypes on chrs. without haplotypes')
        self.phenotype_df = phenotype_df[m]
        self.phenotype_pos_df = phenotype_pos_df[m]

    def _calculate_cis_ranges(self):
        # check phenotypes & calculate genotype ranges
        # get genotype indexes corresponding to cis-window of each phenotype
        self.chr_variant_dfs = {c:g[['pos', 'index']] for c,g in self.variant_df.groupby('chrom')}
        self.cis_ranges, drop_ids = get_cis_ranges(self.phenotype_pos_df, self.chr_variant_dfs, self.window)
        if len(drop_ids) > 0:
            print(f"    ** dropping {len(drop_ids)} phenotypes without variants in cis-window")
            self.phenotype_df = self.phenotype_df.drop(drop_ids)
            self.phenotype_pos_df = self.phenotype_pos_df.drop(drop_ids)
        if 'pos' in self.phenotype_pos_df:
            self.phenotype_start = self.phenotype_pos_df['pos'].to_dict()
            self.phenotype_end = self.phenotype_start
        else:
            self.phenotype_start = self.phenotype_pos_df['start'].to_dict()
            self.phenotype_end = self.phenotype_pos_df['end'].to_dict()
        self.n_phenotypes = self.phenotype_df.shape[0]

        if self.group_s is not None:
            self.group_s = self.group_s.loc[self.phenotype_df.index].copy()
            self.n_groups = self.group_s.unique().shape[0]

    @background(max_prefetch=6)
    def generate_data(self, chrom=None, verbose=False):
        """
        Generate batches from genotype data

        Returns: phenotype array, genotype matrix, genotype index, phenotype ID(s), [group ID]
        """
        if chrom is None:
            phenotype_ids = self.phenotype_df.index
            chr_offset = 0
        else:
            phenotype_ids = self.phenotype_pos_df[self.phenotype_pos_df['chr'] == chrom].index
            if self.group_s is None:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'], return_index=True))}
            else:
                offset_dict = {i:j for i,j in zip(*np.unique(self.phenotype_pos_df['chr'][self.group_s.drop_duplicates().index], return_index=True))}
            chr_offset = offset_dict[chrom]

        index_dict = {j:i for i,j in enumerate(self.phenotype_df.index)}

        if self.group_s is None:
            for k,phenotype_id in enumerate(phenotype_ids, chr_offset+1):
                if verbose:
                    print_progress(k, self.n_phenotypes, 'phenotype')
                p = self.phenotype_df.values[index_dict[phenotype_id]]
                # p = self.phenotype_df.values[k]
                r = self.cis_ranges[phenotype_id]
                yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), phenotype_id
        else:
            gdf = self.group_s[phenotype_ids].groupby(self.group_s, sort=False)
            for k,(group_id,g) in enumerate(gdf, chr_offset+1):
                if verbose:
                    print_progress(k, self.n_groups, 'phenotype group')
                # check that ranges are the same for all phenotypes within group
                assert np.all([self.cis_ranges[g.index[0]][0] == self.cis_ranges[i][0] and self.cis_ranges[g.index[0]][1] == self.cis_ranges[i][1] for i in g.index[1:]])
                group_phenotype_ids = g.index.tolist()
                # p = self.phenotype_df.loc[group_phenotype_ids].values
                p = self.phenotype_df.values[[index_dict[i] for i in group_phenotype_ids]]
                r = self.cis_ranges[g.index[0]]
                yield p, self.genotype_df.values[r[0]:r[-1]+1], np.arange(r[0],r[-1]+1), group_phenotype_ids, group_id
