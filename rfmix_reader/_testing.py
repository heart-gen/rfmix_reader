from rfmix_reader import read_rfmix, interpolate_array

def _load_genotypes(plink_prefix_path):
    from tensorqtl import pgen
    pgr = pgen.PgenReader(plink_prefix_path)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df


def _load_admix(prefix_path, binary_dir):
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def __testing__():
    basename = "/projects/b1213/large_projects/brain_coloc_app/input"
    # Local ancestry
    prefix_path = f"{basename}/local_ancestry_rfmix/_m/"
    binary_dir = f"{basename}/local_ancestry_rfmix/_m/binary_files/"
    loci, rf_q, admix = _load_admix(prefix_path, binary_dir)
    loci.rename(columns={"chromosome": "chrom",
                         "physical_position": "pos"},
                inplace=True)
    # Variant data
    plink_prefix = f"{basename}/genotypes/TOPMed_LIBD"
    _, variant_df = _load_genotypes(plink_prefix)
    variant_df = variant_df.drop_duplicates(subset=["chrom","pos"],keep='first')
    variant_loci_df = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
                                       how="outer", indicator=True)\
                                .loc[:, ["chrom", "pos", "i", "_merge"]]
    data_path = f"{basename}/local_ancestry_rfmix/_m"
    z = interpolate_array(variant_loci_df, admix, data_path)
    #arr_geno = np.array(variant_loci_df[~(variant_loci_df["_merge"] == "right_only")].index)
    #new_admix = z[arr_geno, :]
    return variant_loci_df, z
