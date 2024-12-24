"""
Script to imputate loci to genotype
"""
import zarr

def _load_genotypes(plink_prefix_path):
    from tensorqtl import pgen
    pgr = pgen.PgenReader(plink_prefix_path)
    variant_df = pgr.variant_df
    variant_df.loc[:, "chrom"] = "chr" + variant_df.chrom
    return pgr.load_genotypes(), variant_df


def _load_admix():
    from rfmix_reader import read_rfmix
    prefix_path = "/projects/b1213/large_projects/brain_coloc_app/"+\
        "input/local_ancestry_rfmix/_m/"
    binary_dir = "/projects/b1213/large_projects/brain_coloc_app/"+\
        "input/local_ancestry_rfmix/_m/binary_files/"
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def _expand_array(dx, admix):
    import numpy as np
    z = zarr.zeros((dx.shape[0], admix.shape[1]),
                   chunks=(1000, 100), dtype='float32')
    # Fill with NaNs
    arr_nans = np.array(dx.loc[dx.isnull().any(axis=1)].index,
                        dtype=np.int32)
    z[arr_nans, :] = np.nan
    rm(arr_nans)
    # Fill with local ancestry
    arr = np.array(dx.dropna().index)
    z[arr, :] = admix.compute()
    return z


def testing():
    # Local ancestry
    loci, rf_q, admix = _load_admix()
    loci.rename(columns={"chromosome": "chrom",
                         "physical_position": "pos"},
                inplace=True)
    sample_ids = list(rf_q.sample_id.unique().to_pandas())
    # Variant data
    plink_prefix = "/projects/b1213/large_projects/brain_coloc_app/"+\
        "input/genotypes/TOPMed_LIBD"
    _, variant_df = _load_genotypes(plink_prefix)
    variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
                                            keep='first')
    dx = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
                          how="outer", indicator=True)\
                   .loc[:, ["chrom", "pos", "i"]]
    z = _expand_array(dx, admix)
