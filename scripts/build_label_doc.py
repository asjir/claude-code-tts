#!/usr/bin/env python3
"""Build the markdown labeling document from the saved-session corpus.

Walks one or more Claude Code session transcripts and emits, per message
chain, a heading + the human prompt as a blockquote + a markdown table of
`input | output | id` rows — one row per assistant turn. The `output` column
is left empty for a human (and later Claude) to fill in; see the labeling
protocol in the README / plan.

Two files are written into `datasets/`:
  - labels.md         human-facing markdown, edited in Obsidian
  - labels_raw.jsonl  one record per row {id, session_id, chain_id,
                      turn_index, prompt_text, assistant_text, is_final},
                      the source of truth for export (the markdown flattens
                      newlines/pipes for rendering and must never be the only
                      copy of a turn's text).

No model calls. Scope mirrors build_dataset.py (positional project query or
--sessions FILE ...), plus --all to sweep every ~/.claude/projects/*/*.jsonl.

Examples:
    # Every project (the full reviewed corpus)
    uv run python scripts/build_label_doc.py --all

    # Just the tts project's sessions
    uv run python scripts/build_label_doc.py claude-code-tts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_dataset import PROJECTS_ROOT, iter_chains, resolve_sessions

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MD = REPO_DIR / "datasets" / "labels.md"
DEFAULT_RAW = REPO_DIR / "datasets" / "labels_raw.jsonl"

# Strip this from project dir names for a readable header label. Project dirs
# encode the cwd with '/' -> '-', e.g. "-Users-me-code-claude-code-tts".
_HOME_PREFIX = str(Path.home()).replace("/", "-")


def chain_slug(chain_id: str, used: set[str]) -> str:
    """A doc-unique short namespace for a chain instance.

    Normally `chain_id[:6]`, but `chain_id` is sha1(prompt), so repeated
    prompts (across or within sessions) collide; a `.N` suffix disambiguates
    so every row id below stays globally unique within the document."""
    base = chain_id[:6]
    slug = base
    n = 1
    while slug in used:
        n += 1
        slug = f"{base}.{n}"
    used.add(slug)
    return slug


def turn_id(slug: str, turn_index: int) -> str:
    """Row key linking a markdown row to its raw turn: `{slug}-{NN}`."""
    return f"{slug}-{turn_index:02d}"


def flatten(text: str) -> str:
    """Make assistant text safe for a single markdown table cell.

    Escapes pipes and folds newlines to <br> so multi-line / code-laden turns
    stay on one row. Code won't render, but every character is preserved."""
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def project_label(project_dir: str) -> str:
    """Readable project name from a ~/.claude/projects dir name."""
    label = project_dir
    if label.startswith(_HOME_PREFIX):
        label = label[len(_HOME_PREFIX) :].lstrip("-")
    return label or project_dir


def resolve_scope(args: argparse.Namespace) -> list[Path]:
    """Resolve session files to scan, oldest first."""
    if args.all:
        files = sorted(PROJECTS_ROOT.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            sys.exit(f"[labels] no .jsonl files under {PROJECTS_ROOT}/*/")
        return files
    return resolve_sessions(args)


def build(paths: list[Path], md_path: Path, raw_path: Path) -> tuple[int, int]:
    """Write labels.md + labels_raw.jsonl; return (n_chains, n_turns)."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    n_chains = n_turns = 0
    used_slugs: set[str] = set()

    with (
        md_path.open("w", encoding="utf-8") as md,
        raw_path.open("w", encoding="utf-8") as raw,
    ):
        md.write("# TTS summarizer labels\n\n")
        md.write(
            "Edit the **output** column. *Italic* = unconfirmed (Claude will "
            "redo it; italic text reading like an instruction is taken as a "
            "hint). Plain non-empty = accepted gold (frozen). Convergence = no "
            "empty and no italic cells.\n\n"
        )
        for path in paths:
            session_id = path.stem
            label = project_label(path.parent.name)
            for chain_id, prompt_text, turns in iter_chains(path):
                n_chains += 1
                slug = chain_slug(chain_id, used_slugs)
                md.write(f"## chain {chain_id} — sess {session_id[:8]}… ({label})\n")
                md.write(f"> User asked: {flatten(prompt_text)}\n\n")
                md.write("| input | output | id |\n|---|---|---|\n")
                for turn_index, text, is_final in turns:
                    n_turns += 1
                    rid = turn_id(slug, turn_index)
                    tag = "[FINAL REPLY]" if is_final else "[PROGRESS UPDATE]"
                    cell = flatten(f"{tag} {text}")
                    md.write(f"| {cell} |  | {rid} |\n")
                    raw.write(
                        json.dumps(
                            {
                                "id": rid,
                                "chain_key": slug,
                                "session_id": session_id,
                                "chain_id": chain_id,
                                "turn_index": turn_index,
                                "prompt_text": prompt_text,
                                "assistant_text": text,
                                "is_final": is_final,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                md.write("\n")
    return n_chains, n_turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        help="Substring scoping which project under ~/.claude/projects to scan.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        metavar="FILE",
        help="Explicit session JSONL files to scan (overrides query).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sweep every ~/.claude/projects/*/*.jsonl (the full corpus).",
    )
    parser.add_argument(
        "--md", type=Path, default=DEFAULT_MD, help=f"Output markdown ({DEFAULT_MD})."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW,
        help=f"Output raw sidecar JSONL ({DEFAULT_RAW}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and not args.query and not args.sessions:
        sys.exit("[labels] pass a project query, --sessions FILE ..., or --all")
    paths = resolve_scope(args)
    n_chains, n_turns = build(paths, args.md, args.raw)
    print(
        f"[labels] {n_chains} chains, {n_turns} turns -> {args.md} (+ {args.raw})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
