#!/usr/bin/env python3
"""Claude's italic-aware round-trip over the labeling document.

The labeling protocol (see build_label_doc.py / the plan) is a self-converging
markdown loop. Each output cell is in one of three states:

  - empty            -> Claude fills it with an *italic* proposal
  - *italic*         -> unconfirmed; Claude redoes it. Italic text that reads
                        like an instruction (e.g. *shorter, drop the file name*)
                        is taken as a hint for the redo.
  - plain, non-empty -> accepted gold; never touched (this covers the user's
                        hand-written seed/anchor cells too).

Because good proposals need a capable model, this is a two-step round-trip that
the agent drives:

    1) uv run scripts/fill_labels.py extract
       -> writes datasets/labels_work.json: every empty/italic cell plus, per
          chain, the already-accepted plain cells as in-context style anchors.
    2) the agent reads that file and writes datasets/labels_filled.json,
       a flat {id: "proposed spoken summary"} map honoring the summarizer rules.
    3) uv run scripts/fill_labels.py apply datasets/labels_filled.json
       -> rewrites only those cells, each wrapped in *italics*. Plain cells are
          left byte-identical; a row that turned plain since extract is skipped.

Convergence = no empty and no italic cells remain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from build_label_doc import DEFAULT_MD, DEFAULT_RAW, flatten
from summarizer import SYSTEM_PROMPT

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORK = REPO_DIR / "datasets" / "labels_work.json"

# A row id is `{slug}-{NN}`; slug is chain_id[:6] with an optional .N suffix.
ID_RE = re.compile(r"^[0-9a-f]{6}(?:\.\d+)?-\d{2}$")
# Split a table row on pipes that aren't escaped as \| inside a cell.
_PIPE_RE = re.compile(r"(?<!\\)\|")


def split_segments(line: str) -> list[str] | None:
    """Return the raw pipe-delimited segments of a table line, or None.

    Segments keep their original spacing/escapes; segments[0] and [-1] are the
    empty strings outside the leading/trailing pipes, so '|'.join() round-trips
    the line byte-for-byte."""
    s = line.rstrip("\n")
    if not s.lstrip().startswith("|"):
        return None
    parts = _PIPE_RE.split(s)
    if len(parts) != 5:  # '', input, output, id, ''
        return None
    return parts


def parse_row(line: str) -> tuple[str, str] | None:
    """If `line` is a data row, return (id, raw_output_segment); else None."""
    segs = split_segments(line)
    if segs is None:
        return None
    rid = segs[3].strip()
    if not ID_RE.match(rid):
        return None
    return rid, segs[2]


def classify(cell: str) -> tuple[str, str]:
    """Classify an output cell -> (state, hint).

    state in {empty, italic, plain}; hint is the inner text for italic cells
    (a redo instruction if the user wrote one), else ""."""
    s = cell.strip()
    if not s:
        return "empty", ""
    if len(s) >= 2 and s.startswith("*") and s.endswith("*") and not s.startswith("**"):
        return "italic", s[1:-1].strip()
    return "plain", ""


def parse_doc(md_path: Path) -> list[tuple[str, str, str]]:
    """Parse labels.md into ordered (id, state, hint) per data row."""
    rows: list[tuple[str, str, str]] = []
    for line in md_path.read_text(encoding="utf-8").splitlines(keepends=True):
        parsed = parse_row(line)
        if parsed is None:
            continue
        rid, out = parsed
        state, hint = classify(out)
        rows.append((rid, state, hint))
    return rows


def load_raw(raw_path: Path) -> dict[str, dict]:
    """Index the raw sidecar by row id."""
    index: dict[str, dict] = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            index[rec["id"]] = rec
    return index


def slug_of(rid: str) -> str:
    """The chain namespace of a row id (everything before the `-NN`)."""
    return rid.rsplit("-", 1)[0]


def extract(md_path: Path, raw_path: Path, out_path: Path) -> int:
    """Write the work-item JSON for the agent to fill. Returns item count."""
    rows = parse_doc(md_path)
    raw = load_raw(raw_path)

    # Per chain: accepted plain cells (anchors) and the curated outputs by id.
    accepted: dict[str, list[str]] = {}  # slug -> [row id]
    plain_text: dict[str, str] = {}  # row id -> accepted output text
    for rid, state, _hint in rows:
        if state == "plain":
            accepted.setdefault(slug_of(rid), []).append(rid)

    # Re-read plain cell text (parse_doc dropped it); cheap second pass.
    for line in md_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_row(line)
        if parsed is None:
            continue
        rid, out = parsed
        state, _ = classify(out)
        if state == "plain":
            plain_text[rid] = out.strip()

    items = []
    for rid, state, hint in rows:
        if state == "plain":
            continue
        rec = raw.get(rid)
        if rec is None:
            print(
                f"[fill] warning: {rid} not in raw sidecar, skipping", file=sys.stderr
            )
            continue
        anchors = [
            {
                "id": aid,
                "is_final": raw[aid]["is_final"],
                "assistant_text": raw[aid]["assistant_text"],
                "output": plain_text[aid],
            }
            for aid in accepted.get(slug_of(rid), [])
            if aid in raw
        ]
        items.append(
            {
                "id": rid,
                "state": state,
                "hint": hint,
                "is_final": rec["is_final"],
                "prompt_text": rec["prompt_text"],
                "assistant_text": rec["assistant_text"],
                "anchors": anchors,
            }
        )

    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "instructions": (
            "For each item, write a one/two-sentence spoken summary of "
            "assistant_text following system_prompt. Use the chain's anchors as "
            "style examples. `hint` is the cell's current italic text: it is "
            "either a user instruction (e.g. 'shorter, drop the file name') — "
            "follow it — or a prior auto-proposal — in which case ignore it and "
            "take a fresh swing at different phrasing. Return a flat JSON map "
            "{id: summary} as labels_filled.json."
        ),
        "items": items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    return len(items)


def apply(md_path: Path, filled_path: Path) -> tuple[int, int]:
    """Rewrite work cells from a {id: text} map. Returns (written, skipped)."""
    filled = json.loads(filled_path.read_text(encoding="utf-8"))
    if not isinstance(filled, dict):
        sys.exit(f"[fill] {filled_path} must be a flat JSON object {{id: text}}")

    out_lines: list[str] = []
    written = skipped = 0
    for line in md_path.read_text(encoding="utf-8").splitlines(keepends=True):
        parsed = parse_row(line)
        if parsed is not None:
            rid, out = parsed
            if rid in filled:
                state, _ = classify(out)
                if state == "plain":
                    # Accepted since extract — never overwrite gold.
                    print(f"[fill] {rid} is accepted, leaving frozen", file=sys.stderr)
                    skipped += 1
                else:
                    segs = split_segments(line)
                    assert segs is not None
                    text = str(filled[rid]).strip()
                    segs[2] = f" *{flatten(text)}* "
                    newline = "\n" if line.endswith("\n") else ""
                    line = "|".join(segs) + newline
                    written += 1
        out_lines.append(line)

    md_path.write_text("".join(out_lines), encoding="utf-8")
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="write work-item JSON for the agent")
    ex.add_argument("-o", "--output", type=Path, default=DEFAULT_WORK)

    ap = sub.add_parser("apply", help="apply a filled {id: text} JSON to labels.md")
    ap.add_argument("filled", type=Path, help="datasets/labels_filled.json")

    args = parser.parse_args()
    if args.cmd == "extract":
        n = extract(args.md, args.raw, args.output)
        print(f"[fill] {n} work items -> {args.output}", file=sys.stderr)
    else:
        written, skipped = apply(args.md, args.filled)
        print(
            f"[fill] applied {written} cells, skipped {skipped} -> {args.md}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
