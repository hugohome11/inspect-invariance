"""Command line: analyse a directory of Inspect logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .matrix import from_eval_logs
from .report import analyse, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect-invariance",
        description=(
            "Test whether the language versions of a benchmark measure the same "
            "thing, and name the items that do not."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    an = sub.add_parser("analyse", help="analyse Inspect eval logs")
    an.add_argument("logs", nargs="+", help="log files, or directories of logs")
    an.add_argument("--reference", "-r", default=None,
                    help="reference language code (default: first found)")
    an.add_argument("--design", "-d", default="epochs", choices=("epochs", "models"),
                    help="what counts as a respondent (default: epochs)")
    an.add_argument("--output", "-o", default=None, help="write the report to a file")

    args = parser.parse_args(argv)

    try:
        matrices = from_eval_logs(args.logs, design=args.design)
        report = render(analyse(matrices, reference=args.reference))
    except (ValueError, OSError) as exc:
        print(f"inspect-invariance: {exc}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
