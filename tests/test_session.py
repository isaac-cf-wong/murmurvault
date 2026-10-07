"""End-to-end test of a recording session with fake audio devices and a fake engine."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from murmurvault import audio, session
from murmurvault.audio import SR
from murmurvault.vault import Segment, Vault


class FakeSource(audio.Source):
    """Feeds 48 kHz stereo blocks of a loud tone with pauses, like a real device would."""

    def __init__(self, *_args, **_kwargs):
        """Accept and ignore the real sources' arguments."""
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self, feed):
        """Feed 2 s of tone then 1 s of silence, repeatedly, from a thread.

        Args:
            feed: Callback receiving the blocks.
        """
        self.running = True

        def loop():
            t = 0
            while self.running:
                n = 4800
                loud = (t // 48000) % 3 != 2
                block = (
                    (0.3 * np.sin(np.arange(t, t + n) * 0.2)).astype(np.float32) if loud else np.zeros(n, np.float32)
                )
                feed(np.stack([block, block], axis=1), 48000)
                t += n
                time.sleep(0.01)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the feeding thread."""
        self.running = False
        if self.thread:
            self.thread.join()


class FakeEngine:
    """Transcribes every chunk as "hello"."""

    name, model = "fake", "fake"

    def transcribe(self, audio_, language=None, fast=False):
        """Return one segment spanning the chunk.

        Args:
            audio_: Samples.
            language: Ignored.
            fast: Ignored.

        Returns:
            One segment.
        """
        return [Segment(0.0, len(audio_) / SR, "hello")]


@pytest.fixture
def fake_devices(monkeypatch):
    """Replace the microphone, system audio and live engine with fakes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(audio, "SoundDeviceSource", FakeSource)
    monkeypatch.setattr(audio, "system_source", lambda *_a: FakeSource())
    monkeypatch.setattr(session, "live_engine", lambda _cfg: FakeEngine())


@pytest.mark.usefixtures("fake_devices")
def test_recording_session_end_to_end(isolated_env):
    """Both tracks are written, and the live draft is saved with per-track speaker labels."""
    cfg = isolated_env
    cfg["live"]["min_chunk_s"], cfg["live"]["max_chunk_s"] = 1.0, 4.0
    vault = Vault(cfg["vault"])
    seen = []
    sess = session.RecordingSession(vault, cfg, title="t", folder="calls", live=True, on_segment=seen.append)
    sess.start()
    time.sleep(1.5)
    rec = vault.get(sess.stop().id)

    assert set(rec.tracks) == {"mic", "system"}
    assert (rec.path / "mic.flac").exists()
    assert (rec.path / "system.flac").exists()
    assert rec.duration > 1.0
    assert rec.folder == "calls"
    assert rec.active_transcript == "live"
    draft = vault.load_transcript(rec)
    assert draft.segments
    assert {s.speaker for s in draft.segments} == {"me", "others"}
    assert len(seen) == len(draft.segments)


@pytest.mark.usefixtures("fake_devices")
def test_failed_start_leaves_no_recording(isolated_env, monkeypatch):
    """If a device fails to open, the empty recording entry is removed."""

    def broken_start(self, feed):
        raise RuntimeError("device busy")

    monkeypatch.setattr(FakeSource, "start", broken_start)
    vault = Vault(isolated_env["vault"])
    sess = session.RecordingSession(vault, isolated_env)
    with pytest.raises(RuntimeError, match="device busy"):
        sess.start()
    assert vault.recordings() == []
