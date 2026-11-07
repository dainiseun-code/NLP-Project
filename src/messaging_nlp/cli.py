"""Public validation and offline scoring; historical model execution is disabled."""

import argparse
import sys
from pathlib import Path

from messaging_nlp import __version__
from messaging_nlp.context import load_public_dataset
from messaging_nlp.public_scope import disabled_execution_message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="messaging-nlp", allow_abbrev=False)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate-public", help="validate public-only JSONL inputs"
    )
    validate.add_argument("public_root", type=Path)
    validate.add_argument("--owner-id", required=True)
    reproduce = commands.add_parser(
        "reproduce",
        help="offline scoring of a retained evidence bundle",
        allow_abbrev=False,
    )
    reproduce.add_argument("--bundle", type=Path, required=True)
    infer = commands.add_parser(
        "infer",
        help="disabled: historical model execution is not supported in this snapshot",
        description=disabled_execution_message("infer"),
        allow_abbrev=False,
    )
    infer.add_argument(
        "--plan", type=Path, help="legacy argument; execution is disabled"
    )
    consent = infer.add_mutually_exclusive_group()
    consent.add_argument(
        "--dry-run", action="store_true", help="legacy flag; execution remains disabled"
    )
    consent.add_argument(
        "--approve-run",
        action="store_true",
        help="legacy flag; cannot enable execution in this snapshot",
    )
    ablation = commands.add_parser(
        "ablation",
        help="disabled: historical ablation execution is not supported in this snapshot",
        description=disabled_execution_message("ablation"),
        allow_abbrev=False,
    )
    ablation.add_argument(
        "--plan", type=Path, help="legacy argument; execution is disabled"
    )
    ablation.add_argument("--dry-run", action="store_true")
    ablation.add_argument("--approve", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"infer", "ablation"}:
        print(disabled_execution_message(args.command), file=sys.stderr)
        return 1
    try:
        if args.command == "reproduce":
            from messaging_nlp.io import canonical_json
            from messaging_nlp.reproduction import reproduce

            print(canonical_json(reproduce(args.bundle)).decode("utf-8"))
            return 0
        dataset = load_public_dataset(args.public_root, owner_id=args.owner_id)
        for request in dataset.requests:
            dataset.bundle(request.message_id)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        label = (
            "invalid public inputs"
            if args.command == "validate-public"
            else "reproduction refused"
            if args.command == "reproduce"
            else "run refused"
        )
        print(f"{label}: {error}", file=sys.stderr)
        return 1
    print("public inputs valid")
    return 0
