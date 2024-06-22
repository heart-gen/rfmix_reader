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
    # Column annotations
    sample_ids = _get_sample_names(rf_q)
    pops = _get_pops(rf_q)
    col_names = [f"{sample}_{pop}" for pop in pops for sample in sample_ids]
    # Intervals to keep
    idx = _find_intervals(admix, len(pops))
    filtered_df = loci[loci["i"].isin(idx)]
    # Extract the physical positions
    physical_positions = filtered_df['physical_position'].to_arrow().tolist()
    # Initialize the BED format list
    bed_format = []
    if physical_positions:
        bed_format.append({
            "chromosome": loci["chromosome"][0],
            "start": loci["physical_position"][0],
            "end": physical_positions[0]
        })
    # Add the subsequent rows
    for i in range(len(physical_positions) - 1):
        bed_format.append({
            "chromosome": filtered_df["chromosome"][idx[i+1]],
            "start": physical_positions[i] + 1,
            "end": physical_positions[i + 1]
        })
    # New Data
    annot_df = DataFrame(bed_format)
    dx = DataFrame(admix[idx,:].compute(),
                   columns=col_names)
    return None


def _get_pops(rf_q: DataFrame):
    return rf_q.drop(["sample_id", "chrom"], axis=1).columns.values


def _get_sample_names(rf_q: DataFrame):
    return rf_q.sample_id.unique().to_arrow()


def _find_intervals(dask_matrix, npops):
    ## This should be done per chromosome!
    from dask.array import diff, where    
    num_cols = round(dask_matrix.shape[1] / npops) if npops == 2 else dask_matrix.shape[1]
    all_indices = set()    
    for col in range(num_cols):
        col_data = dask_matrix[:, col]
        diffs = diff(col_data)
        col_change_indices = where(diffs != 0)[0].compute() # Gets position at change
        all_indices.update(col_change_indices)    
    return sorted(all_indices)


def _testing():
    from rfmix_reader import read_rfmix
    # prefix_path = "../examples/two_populations/out/"
    prefix_path = "/dcs05/lieber/hanlab/jbenjami/projects/"+\
        "localQTL_manuscript/local_ancestry_rfmix/_m/"
    loci, rf_q, admix = read_rfmix(prefix_path, verbose=True)

