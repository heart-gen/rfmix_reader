"""
Import hygiene: the readers must import quietly, without torch, and without
the heavy optional dependencies of the processing / io layers.
"""
import subprocess
import sys

READER_MODULES = [
    "rfmix_reader.readers.read_msp",
    "rfmix_reader.readers.read_rfmix",
    "rfmix_reader.readers.read_flare",
    "rfmix_reader.readers.read_simu",
]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True, text=True, timeout=120,
    )


def test_readers_import_silently():
    code = "import " + ", ".join(READER_MODULES)
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
    assert "Warning" not in proc.stderr, proc.stderr


def test_readers_import_without_torch():
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['torch.cuda'] = None\n"
        "import " + ", ".join(READER_MODULES)
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_fb_reader_imports_without_zarr_xarray():
    code = (
        "import sys; sys.modules['zarr'] = None; sys.modules['xarray'] = None\n"
        "import rfmix_reader.readers.read_rfmix, rfmix_reader.readers.read_msp\n"
        "import rfmix_reader.io\n"
        "from rfmix_reader.io import Chunk, BinaryFileNotFoundError\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_prepare_reference_import_without_bio2zarr():
    code = (
        "import sys; sys.modules['bio2zarr'] = None; sys.modules['bio2zarr.vcf'] = None\n"
        "import rfmix_reader.io.prepare_reference as pr\n"
        "try:\n"
        "    pr.convert_vcf_to_zarr('x.vcf.gz', 'x.zarr', verbose=False)\n"
        "except ImportError as e:\n"
        "    assert 'rfmix-reader[reference]' in str(e), str(e)\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
