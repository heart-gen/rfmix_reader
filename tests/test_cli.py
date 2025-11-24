import sys
import pytest

import rfmix_reader.cli.create_binaries as cli


# ---------------------------
# CLI TESTS
# ---------------------------

def test_cli_invokes_create_binaries(monkeypatch):
    called = {}

    def fake_create_binaries(file_path, binary_dir):
        called["args"] = (file_path, binary_dir)

    monkeypatch.setattr(cli, "create_binaries", fake_create_binaries)

    test_args = ["prog", "input_dir", "--binary_dir", "bin_out"]
    monkeypatch.setattr(sys, "argv", test_args)

    cli.main()
    assert called["args"] == ("input_dir", "bin_out")


def test_cli_version_flag(monkeypatch, capsys):
    test_args = ["prog", "--version"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "prog" in out or cli.__version__ in out
