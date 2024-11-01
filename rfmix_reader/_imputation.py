import cudf as pd
from tensorqtl import pgen
from functools import lru_cache
from rfmix_reader import read_rfmix

@lru_cache()
def get_test_ids():
    test_ids = "../../localQTL/example/sample_id_to_brnum.tsv"
    return pd.read_csv(test_ids, sep="\t", usecols=[1])


@lru_cache()
def get_genotypes():
    plink_path = "/projects/b1213/large_projects/brain_coloc_app/caudate/genes/_m"
    plink_prefix = f"{plink_path}/protected_data/TOPMed_LIBD"
    pgr = pgen.PgenReader(plink_prefix)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df


@lru_cache()
def get_loci():
    select_samples = list(get_test_ids().BrNum.to_pandas())
    prefix_path = "/projects/b1213/large_projects/brain_coloc_app/input/" +\
        "local_ancestry_rfmix/_m"
    binary_dir = f"{prefix_path}/binary_files"
    return read_rfmix(prefix_path, verbose=True, binary_dir=binary_dir)


def testing():
    genotype_df, variant_df = get_genotypes()
    loci, rf_q, admix = get_loci()
