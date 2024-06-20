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


def _get_sample_names(rf_q: DataFrame):
    sample_ids = rf_q.sample_id.unique().to_arrow()
    pops = rf_q.drop(["sample_id", "chrom"], axis=1).columns.values
    return  [f"{sample}_{pop}" for pop in pops for sample in sample_ids]


def _find_change_indices(dask_matrix):
    from dask.array import diff, where
    num_cols = dask_matrix.shape[1]
    change_indices = {}
    for col in range(num_cols):
        col_data = dask_matrix[:, col]
        diffs = diff(col_data)
        col_change_indices = where(diffs != 0)[0].compute() # Gets position at change
        # Add +1 to get the next position
        change_indices[col] = col_change_indices.tolist()
    return change_indices


def _get_intervals(change_indices):
    all_indices = []
    for indices in change_indices.values():
        all_indices.extend(indices)
    return sorted(set(all_indices))


def _testing():
    from rfmix_reader import read_rfmix
    prefix_path = "../examples/two_populations/out/"
    loci, rf_q, admix = read_rfmix(prefix_path, verbose=True)
