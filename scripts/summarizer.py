"""Local Ollama-based summarizer.

Turns an assistant turn into one or two short sentences suitable for TTS
playback. Maintains a running pydantic-ai message history per Claude Code
user-prompt cycle so Ollama's KV cache can be reused turn-over-turn; the
history is cleared and the model unloaded on each new human prompt.

Talks to Ollama through its OpenAI-compatible endpoint at /v1."""

from __future__ import annotations

import hashlib
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

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

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
    "or what was found. Output the spoken summary only, no preamble. "
    "Your replies should be laconic."
    "You will receive an ongoing conversation between the user and the assistant; "
    "summarize ONLY the latest assistant message, treating earlier turns as context."
)

# Disable Qwen3's <think> chain — we want fast direct summaries, not reasoning.
# Ollama's /v1/chat/completions honors `reasoning_effort` (not `think`).
MODEL_SETTINGS = OpenAIChatModelSettings(
    openai_reasoning_effort="none", max_tokens=200, timeout=15
)
WARM_SETTINGS = OpenAIChatModelSettings(
    openai_reasoning_effort="none", max_tokens=1, timeout=15
)

# The ollama server process we started, if any. Only set when this module
# launches ollama itself, so shutdown never kills a pre-existing server.
_ollama_proc: subprocess.Popen | None = None

_agent: Agent | None = None
_histories: dict[str, list[ModelMessage]] = {}
_pending_prompts: dict[str, str] = {}
# session_id -> chain_id (sha1[:12] of the originating user prompt). Lets the
# log group all turns of one assistant chain by `chain_id` post-hoc.
_chain_ids: dict[str, str] = {}
_state_lock = threading.Lock()
# Serializes calls into the Ollama runtime (agent.run_sync, /api/generate
# unload) so the async rewarm thread can't collide with a real summarize.
_runtime_lock = threading.Lock()

# Idle unload: another client (e.g. the Ollama desktop app) may keep pinging
# the server and resetting its keep_alive timer, so the model would stay
# resident even when we're idle. We track our own activity and force an
# unload after IDLE_TIMEOUT seconds of no calls from this process. 6 minutes
# is one minute past Claude Code's default 5-minute idle window, so on a
# typical setup this fires only when the user has stopped working.
IDLE_TIMEOUT = 360.0
_idle_timer: threading.Timer | None = None
_idle_timer_lock = threading.Lock()


def _chain_id_for(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


def _bump_idle_timer() -> None:
    """Reset the idle-unload countdown. Called after every activity that
    would have kept Ollama's own keep_alive alive."""
    global _idle_timer
    with _idle_timer_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
        _idle_timer = threading.Timer(IDLE_TIMEOUT, _idle_unload)
        _idle_timer.daemon = True
        _idle_timer.start()


def _idle_unload() -> None:
    print(
        f"[tts] idle for {IDLE_TIMEOUT/60:.0f}m, unloading {MODEL}",
        file=sys.stderr,
    )
    _unload_model()


def _cancel_idle_timer() -> None:
    global _idle_timer
    with _idle_timer_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None


def _log_io(
    kind: str,
    payload: dict,
    response: dict | None,
    error: str | None = None,
    *,
    session_id: str | None = None,
    chain_id: str | None = None,
) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "session_id": session_id,
        "chain_id": chain_id,
        "request": payload,
        "response": response,
        "error": error,
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
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


def _get_agent() -> Agent:
    global _agent
    if _agent is not None:
        return _agent
    model = OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"),
    )
    _agent = Agent(model, system_prompt=SYSTEM_PROMPT)
    return _agent


def warm_model() -> None:
    """Warm the model through the same chat-template path used at request time.

    Going through pydantic-ai (rather than /api/generate) means the cached
    KV prefix covers the system prompt + chat scaffolding, so the first real
    summarize call only has to process the new user-message body."""
    print(f"[tts] warming {MODEL}...", file=sys.stderr)
    agent = _get_agent()
    try:
        with _runtime_lock:
            result = agent.run_sync("hi", model_settings=WARM_SETTINGS)
        _log_io("warm", {"prompt": "hi"}, {"output": result.output})
        _bump_idle_timer()
    except Exception as e:
        _log_io("warm", {"prompt": "hi"}, None, error=repr(e))
        raise


def begin_user_prompt(session_id: str, text: str) -> None:
    """Start a new assistant chain for `session_id`.

    Atomically: stamps a chain_id (sha1[:12] of the prompt) so all turns
    of the chain share an id in the log, stores the pending user prompt
    for the first summarize() call, drops prior history, and evicts +
    asynchronously rewarms the Ollama KV cache so the next call hits a
    fresh model. Other sessions' histories are untouched, but they'll
    re-prefill on their next call (model unload is global — there's
    only one Ollama process)."""
    chain_id = _chain_id_for(text)
    with _state_lock:
        _pending_prompts[session_id] = text
        _chain_ids[session_id] = chain_id
        _histories.pop(session_id, None)
    _unload_model(session_id=session_id, chain_id=chain_id)
    _rewarm_async(session_id=session_id, chain_id=chain_id)


def forget_session(session_id: str) -> None:
    """Drop in-memory state for a session without touching the model."""
    with _state_lock:
        _histories.pop(session_id, None)
        _pending_prompts.pop(session_id, None)
        _chain_ids.pop(session_id, None)


def _unload_model(
    *, session_id: str | None = None, chain_id: str | None = None
) -> None:
    payload = {"model": MODEL, "keep_alive": 0}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with _runtime_lock:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
        _log_io("unload", payload, body, session_id=session_id, chain_id=chain_id)
    except Exception as e:
        _log_io(
            "unload",
            payload,
            None,
            error=repr(e),
            session_id=session_id,
            chain_id=chain_id,
        )


def _rewarm_async(
    *, session_id: str | None = None, chain_id: str | None = None
) -> None:
    """Reload the model through the chat-template path so the system-prompt
    KV prefix is cached before the first real summarize call."""

    def _warm():
        agent = _get_agent()
        try:
            with _runtime_lock:
                result = agent.run_sync("hi", model_settings=WARM_SETTINGS)
            _log_io(
                "rewarm",
                {"prompt": "hi"},
                {"output": result.output},
                session_id=session_id,
                chain_id=chain_id,
            )
            _bump_idle_timer()
        except Exception as e:
            _log_io(
                "rewarm",
                {"prompt": "hi"},
                None,
                error=repr(e),
                session_id=session_id,
                chain_id=chain_id,
            )
            print(f"[tts] rewarm failed: {e}", file=sys.stderr)

    threading.Thread(target=_warm, daemon=True).start()


def summarize(session_id: str, assistant_text: str) -> str:
    with _state_lock:
        history = list(_histories.get(session_id, []))
        pending = _pending_prompts.get(session_id)
        chain_id = _chain_ids.get(session_id)
    if not history:
        if pending:
            prompt = f"User asked:\n{pending}\n\nAssistant replied:\n{assistant_text}"
        else:
            prompt = assistant_text
    else:
        prompt = assistant_text

    agent = _get_agent()
    request_payload = {
        "session": session_id,
        "prompt": prompt,
        "history": ModelMessagesTypeAdapter.dump_python(history),
    }
    try:
        with _runtime_lock:
            result = agent.run_sync(
                prompt, message_history=history, model_settings=MODEL_SETTINGS
            )
    except Exception as e:
        _log_io(
            "summarize",
            request_payload,
            None,
            error=repr(e),
            session_id=session_id,
            chain_id=chain_id,
        )
        raise

    new_history = list(result.all_messages())
    with _state_lock:
        _histories[session_id] = new_history
        _pending_prompts.pop(session_id, None)

    output = (result.output or "").strip()
    history_dump = ModelMessagesTypeAdapter.dump_python(new_history)
    _log_io(
        "summarize",
        request_payload,
        {"output": output, "history": history_dump},
        session_id=session_id,
        chain_id=chain_id,
    )
    _bump_idle_timer()
    return output


def verbalize(text: str) -> str:
    """Make dotted tokens read better aloud.

    "4.8" -> "4 point 8" (handles 4.8.2 too); other word.word dots like
    "config.yaml" become spaces so filenames aren't read as one word."""
    text = re.sub(r"(\d)\.(?=\d)", r"\1 point ", text)
    text = re.sub(r"(\w)\.(?=\w)", r"\1 ", text)
    return text


def shutdown() -> None:
    """Stop the ollama server we started, or just unload the model if we
    attached to a pre-existing one (so VRAM is released without disturbing
    other users of the server)."""
    _cancel_idle_timer()
    if _ollama_proc is not None and _ollama_proc.poll() is None:
        print("[tts] stopping ollama server...", file=sys.stderr)
        _ollama_proc.terminate()
        try:
            _ollama_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ollama_proc.kill()
        return
    print(f"[tts] unloading {MODEL}...", file=sys.stderr)
    _unload_model()
