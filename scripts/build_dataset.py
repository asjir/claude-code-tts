#!/usr/bin/env python3
"""Build a summarization dataset by replaying saved Claude Code sessions.

Walks one or more session JSONL transcripts, feeds each assistant message
through the local summarizer (hitting the Ollama server), and writes one
record per message chain in the format `view_dataset.py` reads.

Replaying the saved transcripts — rather than logging live — means the same
corpus can be re-summarized under different prompts/models to compare
behaviour. See the README "Terminology" section for message/chain definitions.

Examples:
    # Dry run: parse + report chains for the tts project, no model calls
    uv run python scripts/build_dataset.py claude-code-tts --dry-run

    # Build a dataset from the same project's sessions
    uv run python scripts/build_dataset.py claude-code-tts -o datasets/base_0.jsonl

    # Build from explicit transcript files, capped at 5 chains
    uv run python scripts/build_dataset.py --sessions a.jsonl b.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import summarizer
from pydantic_ai.messages import ModelMessagesTypeAdapter
from sessions import iter_turns

REPO_DIR = Path(__file__).resolve().parent.parent
PROJECTS_ROOT = Path.home() / ".claude/projects"
DEFAULT_OUTPUT = REPO_DIR / "datasets" / "base_0.jsonl"


def find_project(query: str) -> Path:
    """Find the project dir under PROJECTS_ROOT whose name best matches `query`.

    Mirrors tts_watch.resolve_scope: shortest dir name containing `query`
    (case-insensitive), ties broken by most-recently-modified."""
    q = query.lower()
    candidates = [
        d for d in PROJECTS_ROOT.iterdir() if d.is_dir() and q in d.name.lower()
    ]
    if not candidates:
        sys.exit(f"[dataset] no project under {PROJECTS_ROOT} matches {query!r}")
    candidates.sort(key=lambda d: (len(d.name), -d.stat().st_mtime))
    return candidates[0]


def resolve_sessions(args: argparse.Namespace) -> list[Path]:
    """Resolve the list of session JSONL files to replay, oldest first."""
    if args.sessions:
        paths = [Path(p) for p in args.sessions]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            sys.exit(f"[dataset] session file(s) not found: {missing}")
        return paths
    if not args.query:
        sys.exit("[dataset] pass a project query or --sessions FILE ...")
    project = find_project(args.query)
    print(f"[dataset] scoped to {project.name}", file=sys.stderr)
    files = sorted(project.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"[dataset] no .jsonl files under {project}")
    return files


def _chain_record(
    session_id: str, chain_id: str, n_updates: int, n_finals: int, *, dry_run: bool
) -> dict:
    """Build a dataset record for the chain currently held in summarizer state."""
    record: dict = {
        "chain_id": chain_id,
        "session_id": session_id,
        "n_turns": n_updates + n_finals,
        "n_updates": n_updates,
        "n_finals": n_finals,
    }
    if not dry_run:
        record["model"] = summarizer.MODEL
        messages = summarizer.get_history(session_id)
        record["messages"] = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    return record


def iter_chains(
    path: Path,
) -> Iterator[tuple[str, str, list[tuple[int, str, bool]]]]:
    """Yield (chain_id, prompt_text, turns) per completed chain in a session.

    A chain opens on a human prompt and closes at the next prompt or EOF.
    `chain_id` is the sha1[:12] of the prompt; `turns` is a list of
    (turn_index, assistant_text, is_final), turn_index 0-based within the chain.
    Model-free: shared by the dataset builder and the label-doc builder.

    Assistant turns that appear before any recognized human prompt are skipped.
    Older transcripts predate the `origin.kind == "human"` field that
    `classify_turn` keys on, so none of their prompts are recognized and the
    whole session would otherwise collapse into one promptless blob chain —
    dropping those keeps the corpus to cleanly-delimited recent sessions.
    Empty chains (a prompt with no assistant text before the next prompt) are
    skipped — they'd carry no summaries."""
    chain_id: str | None = None
    prompt_text = ""
    turns: list[tuple[int, str, bool]] = []

    for kind, text, is_final in iter_turns(path):
        if kind == "human_prompt":
            if chain_id is not None and turns:
                yield chain_id, prompt_text, turns
            chain_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            prompt_text = text
            turns = []
        else:  # assistant_text
            if chain_id is None:
                # No preceding human prompt — older/mid-stream transcript; skip.
                continue
            turns.append((len(turns), text, is_final))

    if chain_id is not None and turns:
        yield chain_id, prompt_text, turns


def iter_chain_records(path: Path, *, dry_run: bool) -> Iterator[dict]:
    """Yield one dataset record per completed message chain in a session file.

    Replays each chain (from `iter_chains`) through the summarizer so its
    running history is read off just before the next chain resets it, giving
    each record that chain's full request/response history."""
    session_id = path.stem
    summarizer.forget_session(session_id)

    for chain_id, prompt_text, turns in iter_chains(path):
        n_updates = n_finals = 0
        if not dry_run:
            summarizer.reset_chain(session_id, prompt_text)
        for _turn_index, text, is_final in turns:
            if not dry_run:
                summarizer.summarize(session_id, text, is_final)
            if is_final:
                n_finals += 1
            else:
                n_updates += 1
        yield _chain_record(session_id, chain_id, n_updates, n_finals, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        help="Substring to scope which project under ~/.claude/projects to "
        "replay. Ignored if --sessions is given.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        metavar="FILE",
        help="Explicit session JSONL files to replay (overrides query).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dataset JSONL (default: {DEFAULT_OUTPUT}). Overwritten.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many chains total (0 = no limit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report chains without calling the model or writing output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sessions = resolve_sessions(args)

    if not args.dry_run:
        summarizer.ensure_server()
        summarizer.warm_model()
        args.output.parent.mkdir(parents=True, exist_ok=True)

    out_fh = None if args.dry_run else args.output.open("w", encoding="utf-8")
    total_chains = total_turns = 0
    try:
        for path in sessions:
            print(f"[dataset] {path.name}", file=sys.stderr)
            for record in iter_chain_records(path, dry_run=args.dry_run):
                total_chains += 1
                total_turns += record["n_turns"]
                if out_fh is not None:
                    out_fh.write(
                        json.dumps(record, ensure_ascii=False, default=str) + "\n"
                    )
                if args.dry_run:
                    print(
                        f"  chain {record['chain_id']}  turns={record['n_turns']} "
                        f"(updates={record['n_updates']}, finals={record['n_finals']})"
                    )
                if args.limit and total_chains >= args.limit:
                    raise _Stop
    except _Stop:
        pass
    finally:
        if out_fh is not None:
            out_fh.close()
        if not args.dry_run:
            summarizer.shutdown()

    where = "(dry run, nothing written)" if args.dry_run else str(args.output)
    print(
        f"[dataset] {total_chains} chains, {total_turns} turns -> {where}",
        file=sys.stderr,
    )


class _Stop(Exception):
    """Internal sentinel to break out of the nested replay loop at --limit."""


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
