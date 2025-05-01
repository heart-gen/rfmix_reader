from pathlib import Path
from rfmix_reader import (
    read_rfmix,
    interpolate_array,
    generate_tagore_bed
)

def _load_genotypes(plink_prefix_path):
    from tensorqtl import pgen
    pgr = pgen.PgenReader(plink_prefix_path)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df


def _load_admix(prefix_path, binary_dir):
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def _load_real_data():
    basename = "/projects/b1213/resources/processed-data/local-ancestry"
    prefix_path = Path(basename) / "rfmix-version/_m/"
    binary_dir = Path(prefix_path) / "binary_files/"
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def _load_simu_data(pop=2):
    basename = "/projects/p32505/projects/rfmix_reader-benchmarking/input/simulations"
    pop_loc = "two_populations" if pop == 2 else "three_populations"
    prefix_path = Path(basename) / pop_loc / "_m/rfmix-out/"
    binary_dir = prefix_path / "binary_files"
    if binary_dir.exists():
        return read_rfmix(prefix_path, binary_dir=binary_dir)
    else:
        return read_rfmix(prefix_path, binary_dir=binary_dir,
                          generate_binary=True)


def _testing_simu_viz(pop_num, sample_num):
    loci, rf_q, admix = _load_simu_data(pop_num)
    return generate_tagore_bed(loci, rf_q, admix, sample_num)


def _testing_real_viz(sample_num):
    loci, rf_q, admix = _load_real_data()
    return generate_tagore_bed(loci, rf_q, admix, sample_num)


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
