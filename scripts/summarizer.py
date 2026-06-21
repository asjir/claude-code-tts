"""Local Ollama-based summarizer.

Turns an assistant turn into one or two short sentences suitable for TTS
playback. Maintains a running pydantic-ai message history per Claude Code
user-prompt cycle so Ollama's KV cache can be reused turn-over-turn; the
history is cleared and the model unloaded on each new human prompt.

Talks to Ollama through its OpenAI-compatible endpoint at /v1."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

MODEL = "qwen3.5:2b-mlx"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You turn an assistant's response into one or two short, spoken sentences. "
    "Rules: speak in plain conversational English; never read URLs, file paths, "
    "code, identifiers, or shell commands aloud (say 'a link', 'the config file', "
    "'the function', etc.); skip greetings and meta-talk; just say what changed "
    "or what was found. Output the spoken summary only, no preamble. "
    "Your replies should be laconic."
    "You will receive an ongoing conversation between the user and the assistant; "
    "summarize ONLY the latest assistant message, treating earlier turns as context. "
    "Each message to summarize is tagged [PROGRESS UPDATE] or [FINAL REPLY]. "
    "For a progress update, say what the assistant is doing right now, in the "
    "present tense (e.g. 'Editing the config file'). For a final reply, say what "
    "was accomplished or found, as a wrap-up. Never read the tag aloud.",
    "If the message needs the user to read it, only output the prompt for them to do so.",
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
        f"[tts] idle for {IDLE_TIMEOUT / 60:.0f}m, unloading {MODEL}",
        file=sys.stderr,
    )
    _unload_model()


def _cancel_idle_timer() -> None:
    global _idle_timer
    with _idle_timer_lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None


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
    with _runtime_lock:
        agent.run_sync("hi", model_settings=WARM_SETTINGS)
    _bump_idle_timer()


def reset_chain(session_id: str, text: str) -> None:
    """Start a new chain for `session_id` without cycling the model.

    Stores the pending user prompt for the first summarize() call and drops
    prior history. This is the pure state reset; the live path adds a model
    cycle on top (see begin_user_prompt). The dataset builder uses this
    directly so the model stays resident while replaying many chains."""
    with _state_lock:
        _pending_prompts[session_id] = text
        _histories.pop(session_id, None)


def begin_user_prompt(session_id: str, text: str) -> None:
    """Start a new assistant chain for `session_id` (live path).

    Resets chain state, then evicts + asynchronously rewarms the Ollama KV
    cache so the next call hits a fresh model. Other sessions' histories are
    untouched, but they'll re-prefill on their next call (model unload is
    global — there's only one Ollama process)."""
    reset_chain(session_id, text)
    _unload_model()
    _rewarm_async()


def get_history(session_id: str) -> list[ModelMessage]:
    """Return a copy of the accumulated message history for `session_id`.

    Used by the dataset builder to serialize a finished chain."""
    with _state_lock:
        return list(_histories.get(session_id, []))


def forget_session(session_id: str) -> None:
    """Drop in-memory state for a session without touching the model."""
    with _state_lock:
        _histories.pop(session_id, None)
        _pending_prompts.pop(session_id, None)


def _unload_model() -> None:
    payload = {"model": MODEL, "keep_alive": 0}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with _runtime_lock:
            with urllib.request.urlopen(req, timeout=10):
                pass
    except Exception:
        pass


def _rewarm_async() -> None:
    """Reload the model through the chat-template path so the system-prompt
    KV prefix is cached before the first real summarize call."""

    def _warm():
        agent = _get_agent()
        try:
            with _runtime_lock:
                agent.run_sync("hi", model_settings=WARM_SETTINGS)
            _bump_idle_timer()
        except Exception as e:
            print(f"[tts] rewarm failed: {e}", file=sys.stderr)

    threading.Thread(target=_warm, daemon=True).start()


def _tag(is_final: bool) -> str:
    return "[FINAL REPLY]" if is_final else "[PROGRESS UPDATE]"


def tagged_body(text: str, is_final: bool) -> str:
    """Render an assistant turn as the message body the model summarizes.

    Prefixes the [PROGRESS UPDATE]/[FINAL REPLY] tag the system prompt keys on.
    Shared with the dataset/label tooling so train and serve render identically."""
    return f"{_tag(is_final)}\n{text}"


def first_turn_prompt(pending: str, body: str) -> str:
    """Wrap the first turn's body with the human prompt that opened the chain.

    Only the first turn carries the prompt; later turns rely on message history."""
    return f"User asked:\n{pending}\n\n{body}"


def summarize(session_id: str, assistant_text: str, is_final: bool) -> str:
    """Summarize one assistant message for TTS.

    `is_final` carries the character of the message: True for the concluding
    reply of a turn (stop_reason=end_turn), False for an intermediate progress
    update. It's tagged into the message body so the model can frame the
    summary accordingly (present-tense for progress, wrap-up for final)."""
    with _state_lock:
        history = list(_histories.get(session_id, []))
        pending = _pending_prompts.get(session_id)
    body = tagged_body(assistant_text, is_final)
    if not history and pending:
        prompt = first_turn_prompt(pending, body)
    else:
        prompt = body

    agent = _get_agent()
    with _runtime_lock:
        result = agent.run_sync(
            prompt, message_history=history, model_settings=MODEL_SETTINGS
        )

    new_history = list(result.all_messages())
    with _state_lock:
        _histories[session_id] = new_history
        _pending_prompts.pop(session_id, None)

    output = (result.output or "").strip()
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
