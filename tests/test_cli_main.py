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


def test_cli_export_logical(fb_dir, tmp_path, capsys):
    pytest.importorskip("pyarrow")
    with pytest.raises(SystemExit) as exc:
        cli.main(["convert", "fb", str(fb_dir), str(tmp_path / "cache"), "--keep-posteriors", "--quiet"])
    assert exc.value.code == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(["export-logical", str(tmp_path / "cache"), str(fb_dir / "chr1.vcf.gz"),
                  str(tmp_path / "bundle"), "--max-distance", "1000", "--region", "5030000-5035000",
                  "--sites-per-file", "2", "--quiet"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("bundle/chr1")
    bundle = tmp_path / "bundle" / "chr1"
    assert (bundle / "manifest.json").is_file() and (bundle / "logical_run.R").is_file()
    # 8 records in the region in blocks of 2: rs5 and rs8 are skipped, so four files with 2, 2, 1, 1 sites
    assert (bundle / "sites-3.parquet").is_file() and not (bundle / "sites-4.parquet").exists()
    with pytest.raises(SystemExit) as exc:
        cli.main(["export-logical", str(tmp_path / "cache"), str(fb_dir / "chr1.vcf.gz"),
                  str(tmp_path / "bad"), "--region", "oops", "--quiet"])
    assert exc.value.code == 1
    assert "START-END" in capsys.readouterr().err
