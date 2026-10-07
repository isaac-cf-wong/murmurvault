"""Tests for the OpenAI-compatible transcription engine."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from murmurvault.audio import SR
from murmurvault.engines import OpenAICompatibleEngine, get_engine


def _speech(seconds: float) -> np.ndarray:
    return (0.3 * np.sin(np.arange(int(seconds * SR)))).astype(np.float32)


def test_api_engine_uploads_short_chunks_and_offsets_segments(monkeypatch):
    """A long recording is uploaded in pause-cut chunks, and each chunk's one segment is offset to its start."""
    audio = np.concatenate([_speech(25), np.zeros(SR // 2, np.float32), _speech(20)])  # pause at 25-25.5 s
    uploads = []

    def fake_post(self, chunk, language, response_format):
        uploads.append(len(chunk))
        # Like Qwen3-ASR on oMLX: one segment spanning the whole upload.
        return {"text": "x", "segments": [{"start": 0.0, "end": len(chunk) / SR, "text": f"chunk{len(uploads)}"}]}

    monkeypatch.setattr(OpenAICompatibleEngine, "_post", fake_post)
    segs = OpenAICompatibleEngine("lan", "http://h/v1", "m").transcribe(audio)
    assert len(uploads) == 2
    assert max(uploads) <= 30 * SR
    assert sum(uploads) == len(audio)
    assert 25.0 <= segs[1].start <= 25.5  # second upload starts inside the pause
    assert segs[0].start == 0.0
    assert segs[0].end == pytest.approx(segs[1].start)


def test_api_engine_keeps_a_tail_shorter_than_one_frame(monkeypatch):
    """A cut just before the end does not drop the last few samples; they go with the previous upload."""
    audio = _speech(30.05)
    audio[30 * SR - SR // 10 : 30 * SR] = 0  # the only quiet frame ends at 30.0 s, leaving a 50 ms tail
    uploads = []

    def fake_post(self, chunk, language, response_format):
        uploads.append(len(chunk))
        return {"text": "x", "segments": [{"start": 0.0, "end": len(chunk) / SR, "text": "x"}]}

    monkeypatch.setattr(OpenAICompatibleEngine, "_post", fake_post)
    OpenAICompatibleEngine("lan", "http://h/v1", "m").transcribe(audio)
    assert sum(uploads) == len(audio)


def test_api_engine_chunk_s_from_config():
    """``chunk_s`` in an engine section sets the upload length; it defaults to 30 s."""
    cfg = {"transcription": {"engine": "lan"}, "engines": {"lan": {"type": "openai", "base_url": "u", "model": "m"}}}
    assert get_engine(cfg).chunk_s == 30.0
    cfg["engines"]["lan"]["chunk_s"] = 600
    assert get_engine(cfg).chunk_s == 600.0


def test_api_engine_connection_error_is_runtime_error(monkeypatch):
    """An unreachable server is reported as a RuntimeError, which the CLI shows as a one-line error."""

    def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(RuntimeError, match="cannot reach http://h/v1"):
        OpenAICompatibleEngine("lan", "http://h/v1", "m").transcribe(_speech(1))
