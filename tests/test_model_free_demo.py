"""The public example must stay fictional, model-free and offline."""

import json
import subprocess
import sys
from pathlib import Path


def test_fictional_example_without_models_or_network(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/model_free_demo.py"
    script = """
import importlib.abc
import runpy
import socket
import sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'mlx', 'mlx_lm', 'tokenizers', 'sentencepiece'}:
            raise AssertionError('optional model import: ' + fullname)
sys.meta_path.insert(0, NoModels())
def no_network(*args, **kwargs):
    raise AssertionError('example attempted network access')
socket.create_connection = no_network
socket.socket = no_network
runpy.run_path(sys.argv[1], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(example)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["fictional_demo"] is True
    assert data["measured_study_result"] is False
    assert data["semantic_success"] == "not assessed"
    assert data["technical_status"] == "ok"
    assert data["provider_calls"] == 1
    assert data["retrieved_record_ids"] == ["fictional-contact"]
    assert data["decision"]["selection"]["recipient_contact_ids"] == ["fictional-rowan"]
    assert "seed order" in data["decision"]["draft"]
    assert data["messages_sent"] == 0
    assert list(tmp_path.iterdir()) == []
