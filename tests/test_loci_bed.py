import numpy as np
import pandas as pd
import pytest
import dask.array as da

from rfmix_reader.io.loci_bed import admix_to_bed_individual
from rfmix_reader.readers.read_msp import read_rfmix


def test_admix_to_bed_individual_synthetic():
    """Sample 0 flips [2,0] -> [0,2] at row 5; one change point expected."""
    loci = pd.DataFrame({
        "chromosome": ["chr1"] * 10,
        "physical_position": np.arange(100, 1100, 100),
        "i": np.arange(10),
    })
    g_anc = pd.DataFrame({"sample_id": ["S1", "S2"], "AFR": [0.5, 0.5],
                          "EUR": [0.5, 0.5], "chrom": ["chr1"] * 2})
    admix = np.zeros((10, 2, 2), dtype=np.int8)
    admix[:5, 0, :] = [2, 0]
    admix[5:, 0, :] = [0, 2]
    admix[:, 1, :] = [1, 1]

    bed = admix_to_bed_individual(loci, g_anc, da.from_array(admix), 0,
                                  min_segment=2, verbose=False)
    assert list(bed.columns) == ["chromosome", "start", "end", "S1_AFR", "S1_EUR"]
    assert (bed["chromosome"] == "chr1").all()
    # the change point at row 5 (pos 600) is reported with the new state
    row = bed.loc[bed["end"] == 600].iloc[0]
    assert (row["S1_AFR"], row["S1_EUR"]) == (0, 2)
    first = bed.iloc[0]
    assert (first["S1_AFR"], first["S1_EUR"]) == (2, 0)


def test_admix_to_bed_individual_from_reader(msp_dir):
    loci, g_anc, admix = read_rfmix(str(msp_dir), verbose=False)
    bed = admix_to_bed_individual(loci, g_anc, admix, 0, min_segment=1, verbose=False)

    assert list(bed.columns) == ["chromosome", "start", "end", "Sample_1_EUR", "Sample_1_AFR"]
    chr1 = bed[bed["chromosome"] == "chr1"].sort_values("start")
    # Sample_1 on chr1: EUR/EUR, EUR/AFR, AFR/AFR, AFR/AFR
    np.testing.assert_array_equal(
        chr1[["Sample_1_EUR", "Sample_1_AFR"]].to_numpy(),
        [[2, 0], [1, 1], [0, 2], [0, 2]],
    )
    chr2 = bed[bed["chromosome"] == "chr2"]
    assert (chr2["Sample_1_EUR"] == 2).all() and (chr2["Sample_1_AFR"] == 0).all()


def test_admix_to_bed_individual_bad_sample(msp_dir):
    loci, g_anc, admix = read_rfmix(str(msp_dir), verbose=False)
    with pytest.raises(IndexError):
        admix_to_bed_individual(loci, g_anc, admix, 5, verbose=False)
