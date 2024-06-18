import numpy as np
from dask.array import Array

try:
    from torch.cuda import is_available
except ModuleNotFoundError as e:
    print("Warning: PyTorch is not installed. Using CPU!")
    def is_available():
        return False


if is_available():
    from cudf import DataFrame, read_csv, concat
else:
    from pandas import DataFrame, read_csv, concat

__all__ = ["convert_loci"]

def convert_loci(loci: DataFrame, rf_q: DataFrame, admix: Array):
    admix = DataFrame(admix.compute(),
                      columns=get_sample_names(rf_q))
    return None


def get_sample_names(rf_q: DataFrame):
    sample_ids = rf_q.sample_id.unique().to_arrow()
    pops = rf_q.drop(["sample_id", "chrom"], axis=1).columns.values
    return  [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
