import pytest
import rfmix_reader._constants as consts

# ---------------------------
# CONSTANTS TESTS
# ---------------------------

def test_coordinates_and_sizes():
    coords = consts.COORDINATES
    sizes = consts.CHROM_SIZES

    # Coordinates contain chr1 and X
    assert "1" in coords
    assert "X" in coords
    assert "cx" in coords["1"]

    # Sizes return dicts for both assemblies
    hg37 = sizes.get_sizes("hg37")
    hg38 = sizes.get_sizes("hg38")
    assert isinstance(hg37, dict) and "1" in hg37
    assert isinstance(hg38, dict) and "22" in hg38

    # Assemblies property works
    assert set(sizes.available_assemblies) >= {"hg37", "hg38"}
