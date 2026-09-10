"""
Import hygiene: the core imports quietly, without torch, and the removed
legacy names raise a helpful error.
"""
import subprocess
import sys

import pytest

MODULES = [
    "rfmix_reader",
    "rfmix_reader.core",
    "rfmix_reader.formats",
    "rfmix_reader.ops",
    "rfmix_reader.processing.phase",
    "rfmix_reader.cli.main",
]


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True, text=True, timeout=120,
    )


def test_modules_import_silently():
    proc = _run("import " + ", ".join(MODULES))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
    assert "Warning" not in proc.stderr, proc.stderr


def test_modules_import_without_torch_or_cupy():
    code = (
        "import sys\n"
        "for m in ('torch', 'torch.cuda', 'cupy', 'cudf'): sys.modules[m] = None\n"
        "import " + ", ".join(MODULES) + "\n"
        "from rfmix_reader.backends import use_gpu, describe_gpus\n"
        "assert use_gpu() is False and describe_gpus() == []\n"
    )
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr


def test_top_level_import_is_light():
    code = (
        "import sys, rfmix_reader\n"
        "heavy = [m for m in ('xarray', 'zarr', 'dask', 'cyvcf2', 'matplotlib') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
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


@pytest.mark.parametrize("name", ["read_rfmix", "read_rfmix_fb", "write_data", "create_binaries", "Chunk"])
def test_removed_names_point_to_replacement(name):
    import rfmix_reader

    with pytest.raises(AttributeError, match="removed in 0.6"):
        getattr(rfmix_reader, name)
