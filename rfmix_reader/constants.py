"""
Adapted from `main.py` script in the `tagore` package.
Source: https://github.com/jordanlab/tagore/blob/master/src/tagore/main.py
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

__all__ = ["CHROM_SIZES", "COORDINATES"]

# List of (chrom, cx, cy, ht, width)
_COORDINATE: List[Tuple[str, float, float, float, float]] = [
    ("1", 128.6, 1.5, 1654.5, 118.6),
    ("2", 301.4, 43.6, 1612.4, 118.6),
    ("3", 477.6, 341.4, 1314.7, 118.6),
    ("4", 655.6, 517.9, 1138.1, 118.6),
    ("5", 835.4, 461, 1195.1, 118.6),
    ("6", 1012.4, 524.2, 1131.8, 118.6),
    ("7", 1198.2, 608.5, 1047.5, 118.6),
    ("8", 1372.9, 692.8, 963.2, 118.6),
    ("9", 1554.5, 724.4, 931.6, 118.6),
    ("10", 1733.8, 766.6, 889.4, 118.6),
    ("11", 1911.5, 766.6, 889.4, 118.6),
    ("12", 2095.6, 769.7, 886.3, 118.6),
    ("13", 129.3, 2068.8, 766.1, 118.6),
    ("14", 301.6, 2121.5, 713.4, 118.6),
    ("15", 477.5, 2153.1, 681.8, 118.6),
    ("16", 656.7, 2232.2, 602.8, 118.6),
    ("17", 841.2, 2290.7, 544.3, 118.6),
    ("18", 1015.7, 2313.9, 521.1, 118.6),
    ("19", 1199.5, 2437.2, 397.8, 118.6),
    ("20", 1374.4, 2416.1, 418.9, 118.6),
    ("21", 1553, 2510.9, 324.1, 118.6),
    ("22", 1736.9, 2489.8, 345.1, 118.6),
    ("X", 1915.7, 1799.21, 1035.4, 118.6),
    ("Y", 2120.9, 2451.6, 382.7, 118.6),
]

_HG37_SIZES: List[Tuple[str, int]] = [
    ("1", 249250621), ("2", 243199373), ("3", 198022430), ("4", 191154276),
    ("5", 180915260), ("6", 171115067), ("7", 159138663), ("8", 146364022),
    ("9", 141213431), ("10", 135534747), ("11", 135006516), ("12", 133851895),
    ("13", 115169878), ("14", 107349540), ("15", 102531392), ("16", 90354753),
    ("17", 81195210), ("18", 78077248), ("19", 59128983), ("20", 63025520),
    ("21", 48129895), ("22", 51304566), ("X", 155270560), ("Y", 59373566),
]

_HG38_SIZES: List[Tuple[str, int]] = [
    ("1", 248956422), ("2", 242193529), ("3", 198295559), ("4", 190214555),
    ("5", 181538259), ("6", 170805979), ("7", 159345973), ("8", 145138636),
    ("9", 138394717), ("10", 133797422), ("11", 135086622), ("12", 133275309),
    ("13", 114364328), ("14", 107043718), ("15", 101991189), ("16", 90338345),
    ("17", 83257441), ("18", 80373285), ("19", 58617616), ("20", 64444167),
    ("21", 46709983), ("22", 50818468), ("X", 156040895), ("Y", 57227415),
]


@dataclass
class ChromosomeCoordinates:
    """
    Encapsulates chromosome coordinate data.

    Attributes
    ----------
    coordinates : Dict[str, Dict[str, float]]
        A dictionary mapping chromosome name to its coordinate data
        (cx, cy, ht, width).
    """
    _data: List[Tuple[str, float, float, float, float]] = field(
        default_factory=lambda: _COORDINATE,
        init=False,  # Data is provided internally
        repr=False,
        compare=False
    )
    coordinates: Dict[str, Dict[str, float]] = field(init=False)

    def __post_init__(self):
        """Initializes the coordinates dictionary from raw data."""
        self.coordinates = {
            chrom: {"cx": cx, "cy": cy, "ht": ht, "width": width}
            for chrom, cx, cy, ht, width in self._data
        }

@dataclass
class ChromosomeSizes:
    """
    Encapsulates chromosome size data for different genome assemblies.

    Attributes
    ----------
    hg37 : Dict[str, int]
        Chromosome sizes for the hg37 assembly.
    hg38 : Dict[str, int]
        Chromosome sizes for the hg38 assembly.
    """
    hg37: Dict[str, int] = field(default_factory=lambda: dict(_HG37_SIZES), init=False)
    hg38: Dict[str, int] = field(default_factory=lambda: dict(_HG38_SIZES), init=False)

    _all_sizes: Dict[str, Dict[str, int]] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Organizes all sizes into a nested dictionary."""
        self._all_sizes = {
            "hg37": self.hg37,
            "hg38": self.hg38,
        }

    def get_sizes(self, assembly: str) -> Optional[Dict[str, int]]:
        """
        Retrieves chromosome sizes for a specified assembly.

        Parameters
        ----------
        assembly : str
            The name of the genome assembly (e.g., "hg37", "hg38").

        Returns
        -------
        Optional[Dict[str, int]]
            A dictionary mapping chromosome names to their sizes, or None if the
            assembly is not found.
        """
        return self._all_sizes.get(assembly)

    @property
    def available_assemblies(self) -> List[str]:
        """
        Returns a list of available genome assembly names.
        """
        return list(self._all_sizes.keys())


# Instantiate the coordinate data object
COORDINATES = ChromosomeCoordinates().coordinates
CHROM_SIZES = ChromosomeSizes()
