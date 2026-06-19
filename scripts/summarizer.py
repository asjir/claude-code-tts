"""Local Ollama-based summarizer.

Spawns ollama on demand, keeps a small model warm, and turns an assistant
turn into one or two short sentences suitable for TTS playback."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL = "qwen3.5:2b-mlx"
OLLAMA_URL = "http://localhost:11434"

REPO_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_DIR / "logs"
LOG_FILE = LOG_DIR / f"ollama-{datetime.now().strftime('%Y%m%d')}.jsonl"
_log_lock = threading.Lock()

SYSTEM_PROMPT = (
    "You turn an assistant's response into one or two short, spoken sentences. "
    "Rules: speak in plain conversational English; never read URLs, file paths, "
    "code, identifiers, or shell commands aloud (say 'a link', 'the config file', "
    "'the function', etc.); skip greetings and meta-talk; just say what changed "
    "or what was found. Output the spoken summary only, no preamble."
)

# The ollama server process we started, if any. Only set when this module
# launches ollama itself, so shutdown never kills a pre-existing server.
_ollama_proc: subprocess.Popen | None = None


def _log_io(
    kind: str, payload: dict, response: dict | None, error: str | None = None
) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "request": payload,
        "response": response,
        "error": error,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _log_lock:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line)


def ensure_server() -> None:
    global _ollama_proc
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=1).read()
        return
    except Exception:
        pass
    print("[tts] starting ollama server...", file=sys.stderr)
    _ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "OLLAMA_NUM_PARALLEL": "1"},
    )
    for _ in range(40):
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=1).read()
            return
        except Exception:
            time.sleep(0.25)
    sys.exit("[tts] could not reach ollama at localhost:11434")


def warm_model() -> None:
    print(f"[tts] warming {MODEL}...", file=sys.stderr)
    summarize("hello")


def summarize(text: str) -> str:
    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": text,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "num_predict": 200,
            "num_ctx": 2048,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        _log_io("summarize", payload, None, error=repr(e))
        raise
    _log_io("summarize", payload, body)
    return (body.get("response") or "").strip()


def rewarm_async() -> None:
    """Fire-and-forget reload so the next turn finds a warm model.

    Runs in parallel with TTS playback, so latency is hidden as long as
    playback takes longer than model load (~1-2s)."""

    def _warm():
        payload = {
            "model": MODEL,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1, "num_ctx": 2048},
        }
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read())
            _log_io("rewarm", payload, body)
        except Exception as e:
            _log_io("rewarm", payload, None, error=repr(e))
            print(f"[tts] rewarm failed: {e}", file=sys.stderr)

    threading.Thread(target=_warm, daemon=True).start()


def verbalize(text: str) -> str:
    """Make dotted tokens read better aloud.

    "4.8" -> "4 point 8" (handles 4.8.2 too); other word.word dots like
    "config.yaml" become spaces so filenames aren't read as one word."""
    text = re.sub(r"(\d)\.(?=\d)", r"\1 point ", text)
    text = re.sub(r"(\w)\.(?=\w)", r"\1 ", text)
    return text


def shutdown() -> None:
    """Stop the ollama server we started (if any)."""
    if _ollama_proc is not None and _ollama_proc.poll() is None:
        print("[tts] stopping ollama server...", file=sys.stderr)
        _ollama_proc.terminate()
        try:
            _ollama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ollama_proc.kill()
