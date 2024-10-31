from tensorqtl import pgen
from functools import lru_cache

@lru_cache()
def get_genotypes():
    plink_path = "/projects/b1213/large_projects/brain_coloc_app/caudate/genes/_m"
    plink_prefix = f"{plink_path}/protected_data/TOPMed_LIBD"
    pgr = pgen.PgenReader(plink_prefix)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df


def testing():
    genotype_df, variant_df = get_genotypes()
