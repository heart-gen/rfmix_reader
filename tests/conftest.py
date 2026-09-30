from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "data"
LFS_DATA_DIR = REPO_ROOT / "data"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-cuda-tests",
        action="store_true",
        default=False,
        help="Run CUDA integration tests that require additional setup and runtime",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that need the git-LFS chr21 data (minutes, GBs of RAM)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: long-running tests that need the LFS chr21 data"
    )
    config.addinivalue_line("markers", "cuda: tests that need a CUDA GPU")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(30).startswith(b"version https://git-lfs")
    except OSError:
        return True


@pytest.fixture(scope="session")
def chr21_lfs() -> Path:
    """Path to the repository ``data/`` directory, skipping when LFS content is absent."""
    fb = LFS_DATA_DIR / "chr21.fb.tsv"
    if not fb.exists() or _is_lfs_pointer(fb):
        pytest.skip("data/chr21.fb.tsv is missing or is an un-fetched git-LFS pointer")
    return LFS_DATA_DIR


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def msp_dir() -> Path:
    """Directory with two small chromosomes of RFMix ``.msp.tsv`` + ``.rfmix.Q`` files."""
    return FIXTURE_DIR / "msp"


@pytest.fixture(scope="session")
def fb_dir() -> Path:
    """Directory with a small RFMix ``.fb.tsv`` + ``.rfmix.Q`` pair with fractional posteriors."""
    return FIXTURE_DIR / "fb"


@pytest.fixture(scope="session")
def flare_dir() -> Path:
    """Directory with a small FLARE ``.anc.vcf.gz`` (+ ``.tbi``) and ``.global.anc.gz``."""
    return FIXTURE_DIR / "flare"


@pytest.fixture(scope="session")
def simu_dir() -> Path:
    """Directory with a small three-population haptools ``.vcf.gz`` (+ ``.tbi``) and ``.bp``."""
    return FIXTURE_DIR / "simu"
