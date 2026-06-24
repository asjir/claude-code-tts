"""Kokoro + ffplay TTS playback engine.

Loads the Kokoro ONNX model once and streams synthesized audio to ffplay.
A single playback is active at a time: starting a new utterance interrupts
whatever is currently speaking, so the newest turn always wins.

Voice and speed are passed per call rather than read from a module global,
so callers can pick a different voice per session/source."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
KOKORO_MODEL = str(REPO_DIR / "kokoro-v1.0.onnx")
KOKORO_VOICES = str(REPO_DIR / "voices-v1.0.bin")

# A safe default warm voice; the watcher passes real per-session voices to play().
_DEFAULT_WARM_VOICE = "af_bella"

_kokoro = None
_playback_proc: subprocess.Popen | None = None
_playback_thread: threading.Thread | None = None
_playback_lock = threading.Lock()


def load(warm_voice: str = _DEFAULT_WARM_VOICE):
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    print("[tts] loading kokoro model...", file=sys.stderr)
    from kokoro_onnx import Kokoro

    _kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    print("[tts] warming kokoro...", file=sys.stderr)
    _kokoro.create("ok", voice=warm_voice)
    return _kokoro


def play(text: str, voice: str, speed: float) -> None:
    global _playback_thread
    _playback_thread = threading.Thread(
        target=_play_worker, args=(text, voice, speed), daemon=True
    )
    _playback_thread.start()


def stop() -> None:
    """Kill any in-flight ffplay child so a new utterance can take over."""
    global _playback_proc
    with _playback_lock:
        proc = _playback_proc
        _playback_proc = None
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()


def _spawn_ffplay(sample_rate: int) -> subprocess.Popen:
    """Start ffplay reading f32 mono PCM from stdin at sample_rate."""
    return subprocess.Popen(
        [
            "ffplay",
            "-loglevel",
            "quiet",
            "-nodisp",
            "-autoexit",
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ch_layout",
            "mono",
            "-i",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _play_worker(text: str, voice: str, speed: float) -> None:
    async def stream() -> None:
        global _playback_proc
        kokoro = load(voice)
        proc: subprocess.Popen | None = None
        async for samples, sample_rate in kokoro.create_stream(
            text, voice=voice, speed=speed
        ):
            if proc is None:
                # First chunk ready — interrupt any prior playback and start ours.
                stop()
                proc = _spawn_ffplay(sample_rate)
                with _playback_lock:
                    _playback_proc = proc
            try:
                proc.stdin.write(samples.astype("float32").tobytes())
                proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                return
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass

    try:
        asyncio.run(stream())
    except Exception as e:
        print(f"[tts] streaming playback failed: {e}", file=sys.stderr)
