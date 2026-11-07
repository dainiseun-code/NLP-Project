"""Exercise the installed command, never source-file invocation or model runtimes."""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from messaging_nlp import __version__


@pytest.fixture
def cli():
    executable = Path(sysconfig.get_path("scripts")) / "messaging-nlp"
    assert executable.is_file(), "locked installation must provide console entrypoint"

    def invoke(*args):
        return subprocess.run(
            [str(executable), *map(str, args)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    return invoke


@pytest.fixture
def public(tmp_path):
    # No contacts are required to validate a request with genuinely empty context.
    root = tmp_path / "fictional-public"
    root.mkdir()
    rows = {
        "requests": [
            {
                "message_id": "invented",
                "text": "draft smth?? 📨",
                "owner_id": "account",
                "request_time": "2026-06-02T12:00:00Z",
                "time_zone": "UTC",
                "context_snapshot_id": "empty-snapshot",
            }
        ],
        "snapshots": [
            {
                "context_snapshot_id": "empty-snapshot",
                "timestamp": "2026-06-02T12:00:00Z",
                "record_ids": [],
            }
        ],
        "contacts": [],
        "calendar": [],
        "history": [],
    }
    for name, values in rows.items():
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values),
            encoding="utf-8",
        )
    return root


def test_installed_help_and_version(cli):
    help_result = cli("--help")
    assert help_result.returncode == 0
    assert "validate-public" in help_result.stdout
    assert help_result.stderr == ""
    version = cli("--version")
    assert version.returncode == 0
    assert version.stdout.strip() == __version__


def test_installed_validation_success(cli, public):
    result = cli("validate-public", public, "--owner-id", "account")
    assert result.returncode == 0
    assert "public inputs valid" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("kind", ["missing", "malformed", "wrong-owner"])
def test_installed_validation_failure_is_nonzero_without_traceback(cli, public, kind):
    if kind == "missing":
        (public / "history.jsonl").unlink()
    elif kind == "malformed":
        (public / "requests.jsonl").write_text("not JSON\n", encoding="utf-8")
    result = cli(
        "validate-public",
        public,
        "--owner-id",
        "wrong-account" if kind == "wrong-owner" else "account",
    )
    assert result.returncode == 1
    assert "invalid public inputs" in result.stderr
    assert "Traceback" not in result.stderr
    assert "public inputs valid" not in result.stdout


def test_owner_option_required(cli, public):
    assert cli("validate-public", public).returncode == 2


@pytest.mark.parametrize("command", ["score", "train", "evaluate"])
def test_stage_one_has_no_execution_or_scoring_commands(cli, command):
    result = cli(command)
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_infer_is_explicitly_disabled_and_exposes_no_unsafe_overrides(cli):
    missing = cli("infer")
    assert missing.returncode == 1
    assert "infer is disabled" in missing.stderr
    help_result = cli("infer", "--help")
    assert help_result.returncode == 0
    assert "--dry-run" in help_result.stdout
    assert "disabled" in help_result.stdout
    for unsafe in (
        "--skip-restart",
        "--fake-worker",
        "--model",
        "--worker",
        "--approve",
    ):
        assert cli("infer", "--plan", "missing.json", unsafe).returncode == 2


def test_infer_missing_plan_is_safe_nonzero(cli):
    result = cli("infer", "--plan", "does-not-exist.json", "--dry-run")
    assert result.returncode == 1
    assert "infer is disabled" in result.stderr
    assert "Traceback" not in result.stderr


def test_offline_reproduce_requires_bundle_and_exposes_no_model_overrides(cli):
    assert cli("reproduce").returncode == 2
    help_result = cli("reproduce", "--help")
    assert "--bundle" in help_result.stdout
    for unsafe in ("--model", "--review", "--retry", "--bund"):
        assert cli("reproduce", "--bundle", "missing.json", unsafe).returncode == 2


def test_offline_reproduce_missing_bundle_refuses_without_traceback(cli):
    result = cli("reproduce", "--bundle", "does-not-exist.json")
    assert result.returncode == 1
    assert "reproduction refused" in result.stderr
    assert "Traceback" not in result.stderr


def test_disabled_dry_run_never_consults_the_runner(monkeypatch, capsys):
    from messaging_nlp import runner
    from messaging_nlp.cli import main

    def failure(*args, **kwargs):
        raise AssertionError("disabled command must not consult the runner")

    monkeypatch.setattr(runner, "run_plan", failure)
    assert main(["infer", "--plan", "fictional.json", "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert "infer is disabled" in captured.err
    assert "Traceback" not in captured.err


def test_cli_import_and_validation_do_not_import_optional_model_stacks(public):
    # Fresh interpreter with an import blocker: already-imported modules cannot hide
    # accidental runtime coupling. No optional model package is actually imported.
    script = """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'transformers', 'mlx', 'mlx_lm', 'tokenizers', 'sentencepiece'
        }:
            raise AssertionError('optional model import: ' + fullname)
sys.meta_path.insert(0, NoModels())
from messaging_nlp.cli import main
assert main(['validate-public', sys.argv[1], '--owner-id', 'account']) == 0
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(public)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("flags", [[], ["--dry-run"], ["--approve-run"]])
def test_legacy_consent_flags_cannot_enable_cli_execution(monkeypatch, flags, capsys):
    from messaging_nlp import runner
    from messaging_nlp.cli import main

    calls = []

    def fake_run(path, **kwargs):
        calls.append((path, kwargs))
        return {"status": "ready", "execution_authorized": False}

    monkeypatch.setattr(runner, "run_plan", fake_run)
    assert main(["infer", "--plan", "fictional.json", *flags]) == 1
    assert calls == []
    assert "infer is disabled" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [
        ["--approve"],
        ["--appro"],
        ["--skip-restart"],
        ["--model", "fake"],
        ["--worker", "fake"],
        ["--fake-worker"],
        ["--dry-run", "--approve-run"],
    ],
)
def test_pure_consent_cli_rejects_abbreviations_and_unsafe_flags(flags):
    from messaging_nlp.cli import main

    with pytest.raises(SystemExit) as error:
        main(["infer", "--plan", "fictional.json", *flags])
    assert error.value.code == 2


def test_pure_root_parser_rejects_abbreviation():
    from messaging_nlp.cli import main

    with pytest.raises(SystemExit) as error:
        main(["--vers"])
    assert error.value.code == 2
