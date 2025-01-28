from typing import Tuple, List
from zarr import Array as zArray
from dask.array import Array, from_array
from dask.dataframe import from_dask_array, concat
from dask.dataframe import from_pandas as dd_from_pandas

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, from_pandas, Series
else:
    from pandas import DataFrame, Series
    from dask.dataframe import from_pandas

__all__ = ["write_bed", "write_imputed"]

def write_bed(loci: DataFrame, rf_q: DataFrame, admix: Array, outdir: str="./",
              outfile: str = "local-ancestry.bed") -> None:
    admix_ddf = from_dask_array(admix, columns=_get_names(rf_q))
    loci["end"] = loci["pos"] + 1
    loci["hap"] = loci['chrom'].astype(str)+'_'+loci['pos'].astype(str)
    if is_available():
        loci = loci.to_pandas()
    else:
        loci.drop(["i"], axis=1, inplace=True)
    loci_ddf = dd_from_pandas(loci, npartitions=admix_ddf.npartitions)
    bed_ddf = concat([loci_ddf, admix_ddf], axis=1)
    for ii in range(bed_ddf.npartitions):
        chunk = bed_ddf.get_partition(ii).compute()
        chunk.to_csv(f"{outdir}/{outfile}", sep="\t", mode="a",
                     index=False, header=(ii == 0))


def write_imputed(rf_q: DataFrame, admix: Array, variant_loci: DataFrame,
                  z: zArray, outdir: str = "./",
                  outfile: str = "local-ancestry.imputed.bed") -> None:
    loci, admix = _clean_data_imp(admix, variant_loci, z)
    write_bed(loci, rf_q, admix, outfile)


def _get_names(rf_q: DataFrame) -> List[str]:
    if is_available():
        sample_id = list(rf_q.sample_id.unique().to_pandas())
    else:
        sample_id = list(rf_q.sample_id.unique())
    ancestries = list(rf_q.drop(["sample_id", "chrom"], axis=1).columns.values)
    sample_names = [f"{sid}_{anc}" for anc in ancestries for sid in sample_id]
    return sample_names


def _clean_data_imp(admix: Array, variant_loci: DataFrame, z: zArray
                    ) -> Tuple[DataFrame, Array]:
    daz = from_array(z, chunks=admix.chunksize)
    idx_arr = from_array(variant_loci[~(variant_loci["_merge"] ==
                                        "right_only")].index.to_numpy())
    admix_I = daz[idx_arr]
    mask = Series(False, index=variant_loci.index)
    mask.loc[idx_arr] = True
    if is_available():
        variant_loci = from_pandas(variant_loci)
    loci_I = variant_loci[mask].drop(["i", "_merge"], axis=1)\
                               .reset_index(drop=True)
    return loci_I, admix_I
