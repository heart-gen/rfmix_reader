"""
Functions to imputate loci to genotype.

This is a time consuming process, but should only need to be done once.
Loading the data becomes very fast because data is saved to a Zarr.
"""
import zarr

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False

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
    print("Generate empty Zarr!")
    z = zarr.open(f"{path}/local-ancestry.zarr", mode="w",
                  shape=(dx.shape[0], admix.shape[1]),
                  chunks=(1000, 100), dtype='float32')
    # Fill with NaNs
    arr_nans = np.array(dx.loc[dx.isnull().any(axis=1)].index,
                        dtype=np.int32)
    print("Fill Zarr with NANs!")
    z[arr_nans, :] = np.nan
    print("Remove NaN array!")
    del arr_nans
    # Fill with local ancestry
    arr = np.array(dx.dropna().index)
    print("Fill Zarr with data!")
    z[arr, :] = admix.compute()
    return z


def _interpolate_col(col):
    if is_available():
        import cupy as np
    else:
        import numpy as np
    mask = np.isnan(col)
    indices = np.arange(len(col))
    valid = ~mask
    if np.any(valid):
        interpolated = np.interp(indices[mask], indices[valid], col[valid])
        col[mask] = interpolated
    return col


def interpolate_array(dx, admix, path, chunk_size=25000):
    from tqdm import tqdm
    if is_available():
        import cupy as np
    else:
        import numpy as np
    print("Starting expansion!")
    z = _expand_array(dx, admix, path)
    total_rows, _ = z.shape
    # Process the data in chunks
    print("Interpolating data!")
    for i in tqdm(range(0, total_rows, chunk_size),
                  desc="Processing chunks", unit="chunk"):
        end = min(i + chunk_size, total_rows)
        chunk = np.array(z[i:end, :])
        interp_chunk = np.apply_along_axis(_interpolate_col, axis=0, arr=chunk)
        z[i:end, :] = interp_chunk.get() if is_available() else interp_chunk
    return z


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
    data_path = f"{basename}/local_ancestry_rfmix/_m"
    z = interpolate_array(dx, admix, data_path, chunk_size=10000)
