"""Parse Claude Code session JSONL transcripts into talkable turns.

Shared by the live watcher (tts_watch) and the dataset builder so both
classify transcript lines identically. The unit of classification is a
single line; see the README "Terminology" section for message/chain
definitions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def classify_turn(line: str) -> tuple[str, str, bool] | None:
    """Classify a JSONL line as a new human prompt or an assistant text turn.

    Returns ("human_prompt", text, False), ("assistant_text", text, is_final),
    or None. `is_final` is True when stop_reason == "end_turn", marking the
    last assistant message of a turn. Tool-result echoes (also `type: "user"`)
    are filtered out by checking `origin.kind == "human"` and a string content
    body."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    rec_type = rec.get("type")
    msg = rec.get("message") or {}
    if rec_type == "user":
        if (rec.get("origin") or {}).get("kind") != "human":
            return None
        content = msg.get("content")
        if not isinstance(content, str):
            return None
        text = content.strip()
        return ("human_prompt", text, False) if text else None
    if rec_type == "assistant":
        parts = []
        for block in msg.get("content") or []:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        text = "\n".join(p for p in parts if p).strip()
        is_final = msg.get("stop_reason") == "end_turn"
        return ("assistant_text", text, is_final) if text else None
    return None


def classify_codex_turn(line: str) -> tuple[str, str, bool] | None:
    """Classify a Codex rollout `event_msg` line, mirroring `classify_turn`.

    Returns ("human_prompt", text, False), ("assistant_text", text, is_final),
    or None. Codex's `event_msg` stream is the clean analog of Claude's
    transcript: `user_message` is the human prompt (without the AGENTS.md
    preamble that pollutes the raw `response_item`), and `agent_message` is
    assistant text, with `phase == "final_answer"` marking the concluding
    reply (analogous to Claude's stop_reason == "end_turn") vs "commentary"
    for progress updates."""
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None
    if rec.get("type") != "event_msg":
        return None
    p = rec.get("payload") or {}
    p_type = p.get("type")
    if p_type == "user_message":
        text = (p.get("message") or "").strip()
        return ("human_prompt", text, False) if text else None
    if p_type == "agent_message":
        text = (p.get("message") or "").strip()
        is_final = p.get("phase") == "final_answer"
        return ("assistant_text", text, is_final) if text else None
    return None


def iter_turns(path: Path) -> Iterator[tuple[str, str, bool]]:
    """Yield classified turns from a full session JSONL, in file order.

    Unlike the live watcher's `follow()`, this reads the whole file from the
    start — used to replay a completed session for dataset building."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            turn = classify_turn(line)
            if turn is not None:
                yield turn
