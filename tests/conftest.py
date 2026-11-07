"""Mark historical checks unavailable when private study evidence is withheld."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_FILES = (
    "docs/stage1-budget-opening.json",
    "docs/budget-usage-reset-20261027.json",
    "docs/stage6-revision-authorization-20261028.json",
    "data/stage1-v1/evaluator/provenance/annotation-audit/input-access.json",
)
PRIVATE_TESTS = Path(__file__).with_name("private-evidence-tests.txt")


def pytest_collection_modifyitems(items):
    if all((ROOT / name).is_file() for name in PRIVATE_FILES):
        return
    private_tests = {
        line
        for line in PRIVATE_TESTS.read_text().splitlines()
        if line and not line.startswith("#")
    }
    reason = pytest.mark.skip(
        reason="requires historical evidence withheld from the public snapshot"
    )
    for item in items:
        if item.nodeid in private_tests:
            item.add_marker(reason)
