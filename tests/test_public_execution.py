"""Public execution refusal is independent of private plans and optional runtimes."""

import subprocess
import sys
from pathlib import Path

import pytest

from messaging_nlp.cli import main


@pytest.mark.parametrize(
    "command,approval", [("infer", "--approve-run"), ("ablation", "--approve")]
)
@pytest.mark.parametrize("mode", ["plain", "dry-run", "approved"])
def test_cli_refuses_without_reading_plan_or_importing_runner(
    monkeypatch, tmp_path, capsys, command, approval, mode
):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled execution attempted artifact access")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    for name in ("runner", "ablation", "qwen_worker", "preparation_worker"):
        monkeypatch.setitem(sys.modules, f"messaging_nlp.{name}", None)
    args = [command, "--plan", "../../unsafe-or-missing.json"]
    args += (
        ["--dry-run"] if mode == "dry-run" else [approval] if mode == "approved" else []
    )
    assert main(args) == 1
    captured = capsys.readouterr()
    assert f"{command} is disabled" in captured.err
    assert captured.out == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["infer", "ablation"])
@pytest.mark.parametrize(
    "flags",
    [{}, {"dry_run": True}, {"approved": True}, {"dry_run": True, "approved": True}],
)
def test_callable_entrypoints_refuse_before_path_coercion(tmp_path, command, flags):
    from messaging_nlp import ablation, runner

    class NoPathAccess:
        def __fspath__(self):
            raise AssertionError("disabled runner inspected a plan path")

    dry_run = flags.get("dry_run", False)
    approved = flags.get("approved", False)
    with pytest.raises(RuntimeError, match=f"{command} is disabled"):
        if command == "infer":
            runner.run_plan(NoPathAccess(), dry_run=dry_run, approve_run=approved)
        else:
            ablation.run(NoPathAccess(), dry_run=dry_run, approve=approved)
    assert list(tmp_path.iterdir()) == []


def test_fresh_disabled_cli_never_imports_optional_model_stacks():
    script = """
import importlib.abc
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'mlx', 'mlx_lm', 'tokenizers', 'sentencepiece'}:
            raise AssertionError('optional model import: ' + fullname)
        if fullname in {'messaging_nlp.runner', 'messaging_nlp.ablation'}:
            raise AssertionError('historical runner import: ' + fullname)
sys.meta_path.insert(0, NoModels())
from messaging_nlp.cli import main
for command, approval in [('infer', '--approve-run'), ('ablation', '--approve')]:
    assert main([command, '--plan', '/missing/plan.json', approval]) == 1
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
