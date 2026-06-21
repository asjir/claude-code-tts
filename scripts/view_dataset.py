#!/usr/bin/env python3
"""Pretty-print a reassembled-history dataset for human reading.

Loads a reassembled-history JSONL dataset and, for each turn, shows the
message that went in (to be summarized) and the summary that came out.
No model calls — just a readable view of what the summarizer saw and said.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_DIR / "datasets" / "base_0.jsonl"

_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _part(parts: list[dict], kind: str) -> str:
    for p in parts:
        if p.get("part_kind") == kind:
            return p.get("content", "")
    return ""


def _turns(messages: list[dict]):
    """Yield (input, summary) pairs from a request/response message list."""
    requests = [m for m in messages if m.get("kind") == "request"]
    responses = [m for m in messages if m.get("kind") == "response"]
    for req, resp in zip(requests, responses):
        yield _part(req["parts"], "user-prompt"), _part(resp["parts"], "text")


def view(path: Path) -> None:
    with path.open(encoding="utf-8") as fh:
        chains = [json.loads(line) for line in fh if line.strip()]

    for ci, chain in enumerate(chains):
        header = (
            f"chain {chain['chain_id']}  "
            f"session={chain['session_id'] or '-'}  "
            f"turns={chain['n_turns']}"
        )
        print(_c("=" * 70, "90"))
        print(_c(header, "1;36"))
        print(_c("=" * 70, "90"))
        for ti, (msg_in, summary) in enumerate(_turns(chain["messages"]), 1):
            print(_c(f"\n— turn {ti} —", "90"))
            print(_c("IN:", "1;33"))
            print(msg_in)
            print(_c("SUMMARY:", "1;32"))
            print(summary)
        if ci != len(chains) - 1:
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"dataset JSONL to view (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()
    if not args.dataset.exists():
        sys.exit(f"[view] dataset not found: {args.dataset}")
    view(args.dataset)


if __name__ == "__main__":
    main()
