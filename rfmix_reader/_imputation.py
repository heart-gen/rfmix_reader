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


def _load_admix(prefix_path, binary_dir):
    from rfmix_reader import read_rfmix
    return read_rfmix(prefix_path, binary_dir=binary_dir)


def _expand_array(dx, admix, path):
    import numpy as np
    z = zarr.open(f"{path}/local-ancestry.zarr", mode="w",
                  shape=(dx.shape[0], admix.shape[1]),
                  chunks=(1000, 100), dtype='float32')
    # Fill with NaNs
    arr_nans = np.array(dx.loc[dx.isnull().any(axis=1)].index,
                        dtype=np.int32)
    z[arr_nans, :] = np.nan
    rm(arr_nans)
    # Fill with local ancestry
    arr = np.array(dx.dropna().index)
    z[arr, :] = admix.compute()
    return None


def testing():
    basename = "/projects/b1213/large_projects/brain_coloc_app/input"
    # Local ancestry
    prefix_path = f"{basename}/local_ancestry_rfmix/_m/"
    binary_dir = f"{basename}/local_ancestry_rfmix/_m/binary_files/"
    loci, rf_q, admix = _load_admix(prefix_path, binary_dir)
    loci.rename(columns={"chromosome": "chrom",
                         "physical_position": "pos"},
                inplace=True)
    sample_ids = list(rf_q.sample_id.unique().to_pandas())
    # Variant data
    plink_prefix = f"{basename}/genotypes/TOPMed_LIBD"
    _, variant_df = _load_genotypes(plink_prefix)
    variant_df = variant_df.drop_duplicates(subset=["chrom", "pos"],
                                            keep='first')
    dx = variant_df.merge(loci.to_pandas(), on=["chrom", "pos"],
                          how="outer", indicator=True)\
                   .loc[:, ["chrom", "pos", "i"]]
    data_path = f"{basename}/local_ancestry_rfmix/_m/data"
    _expand_array(dx, admix, path)
