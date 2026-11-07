"""Path-safety helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def ensure_output_disjoint(output: Path, protected: Iterable[Path]) -> None:
    """Reject equality or ancestor/descendant overlap after resolving symlinks."""
    resolved_output = Path(output).resolve()
    for path in protected:
        resolved_protected = Path(path).resolve()
        if (
            resolved_output == resolved_protected
            or resolved_protected in resolved_output.parents
            or resolved_output in resolved_protected.parents
        ):
            raise ValueError(f"output overlaps protected path: {resolved_protected}")
