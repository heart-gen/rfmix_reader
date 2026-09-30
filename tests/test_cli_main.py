import pytest

from rfmix_reader.cli import main as cli


def test_cli_convert_and_info(msp_dir, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["convert", "msp", str(msp_dir), str(tmp_path / "cache"), "--quiet"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "chr1.zarr" in out and "chr2.zarr" in out

    with pytest.raises(SystemExit) as exc:
        cli.main(["info", str(tmp_path / "cache")])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "variants:   8" in out and "ancestries: EUR, AFR" in out
    assert "chromosomes: chr1, chr2" in out


def test_cli_errors_exit_1(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["info", str(tmp_path / "missing")])
    assert exc.value.code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert cli.__version__ in capsys.readouterr().out
