#!/usr/bin/env python3
"""Export the converged labeling document to mlx-lm chat finetuning data.

Reads the curated outputs from datasets/labels.md and the exact turns from
datasets/labels_raw.jsonl, and emits one chat conversation per chain in the
mlx-lm `{"messages": [...]}` format. The conversation is assembled exactly the
way summarizer.py builds its live request history — system prompt, then per
turn a user message (tagged body; the first turn also carries the human prompt)
and the assistant's curated summary — so the training prefix matches inference
(teacher forcing: each turn's history holds the *curated* prior summaries).

Requires convergence: every output cell must be plain (no empty, no *italic*);
otherwise it fails loudly. The chains are split by chain (~15% held out) into
train.jsonl / valid.jsonl so no chain leaks across the split.

Examples:
    uv run python scripts/export_finetune.py
    uv run python scripts/export_finetune.py --val-frac 0.15 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from build_label_doc import DEFAULT_MD, DEFAULT_RAW
from fill_labels import classify, load_raw, parse_row, slug_of
from summarizer import SYSTEM_PROMPT, first_turn_prompt, tagged_body

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_DIR / "datasets" / "finetune"


def load_outputs(md_path: Path) -> dict[str, str]:
    """Return {id: curated output}; exit if any cell is empty or italic."""
    outputs: dict[str, str] = {}
    unconverged: list[tuple[str, str]] = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_row(line)
        if parsed is None:
            continue
        rid, cell = parsed
        state, _ = classify(cell)
        if state != "plain":
            unconverged.append((rid, state))
        else:
            outputs[rid] = cell.strip()
    if unconverged:
        sample = ", ".join(f"{rid}({st})" for rid, st in unconverged[:10])
        more = "" if len(unconverged) <= 10 else f" (+{len(unconverged) - 10} more)"
        sys.exit(
            f"[export] {len(unconverged)} cells not converged (empty/italic): "
            f"{sample}{more}\n[export] fill them in before exporting."
        )
    return outputs


def build_conversations(
    raw: dict[str, dict], outputs: dict[str, str]
) -> dict[str, list[dict]]:
    """Assemble one chat conversation per chain, keyed by chain slug."""
    # Group raw turns by chain instance, ordered by turn index.
    chains: dict[str, list[dict]] = {}
    for rec in raw.values():
        chains.setdefault(slug_of(rec["id"]), []).append(rec)

    conversations: dict[str, list[dict]] = {}
    for slug, recs in chains.items():
        recs.sort(key=lambda r: r["turn_index"])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, rec in enumerate(recs):
            body = tagged_body(rec["assistant_text"], rec["is_final"])
            user = first_turn_prompt(rec["prompt_text"], body) if i == 0 else body
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": outputs[rec["id"]]})
        conversations[slug] = messages
    return conversations


def split_and_write(
    conversations: dict[str, list[dict]], out_dir: Path, val_frac: float, seed: int
) -> tuple[int, int]:
    """Split by chain and write train.jsonl / valid.jsonl. Returns (n_tr, n_val)."""
    slugs = sorted(conversations)
    random.Random(seed).shuffle(slugs)
    n_val = math.ceil(len(slugs) * val_frac) if slugs else 0
    val_slugs = set(slugs[:n_val])

    out_dir.mkdir(parents=True, exist_ok=True)
    train_fh = (out_dir / "train.jsonl").open("w", encoding="utf-8")
    valid_fh = (out_dir / "valid.jsonl").open("w", encoding="utf-8")
    try:
        for slug in slugs:
            fh = valid_fh if slug in val_slugs else train_fh
            line = json.dumps({"messages": conversations[slug]}, ensure_ascii=False)
            fh.write(line + "\n")
    finally:
        train_fh.close()
        valid_fh.close()
    return len(slugs) - n_val, n_val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outputs = load_outputs(args.md)
    raw = load_raw(args.raw)
    missing = [rid for rid in raw if rid not in outputs]
    if missing:
        sys.exit(
            f"[export] {len(missing)} raw turns have no md row, e.g. {missing[:5]}"
        )

    conversations = build_conversations(raw, outputs)
    n_tr, n_val = split_and_write(conversations, args.out_dir, args.val_frac, args.seed)
    print(
        f"[export] {n_tr} train + {n_val} valid chains -> {args.out_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
