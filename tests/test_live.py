"""Tests for the live draft transcriber."""

from __future__ import annotations

import numpy as np

from murmurvault.audio import SR
from murmurvault.diarize import assign_speakers
from murmurvault.live import FRAME, LiveTranscriber, find_cut
from murmurvault.vault import Segment


class FakeEngine:
    """Returns one segment per chunk and records the chunk lengths."""

    name, model = "fake", "fake"

    def __init__(self):
        """Start with no calls."""
        self.calls = []

    def transcribe(self, audio, language=None, fast=False):
        """Return one segment starting 0.5 s into the chunk.

        Args:
            audio: Samples.
            language: Ignored.
            fast: Ignored.

        Returns:
            One segment.
        """
        self.calls.append(len(audio))
        return [Segment(0.5, len(audio) / SR, f"chunk{len(self.calls)}")]


def test_find_cut():
    """Chunks end at a pause, or at the quietest frame once the maximum length is reached."""
    quiet = np.zeros(SR * 4, np.float32)
    loud = (0.3 * np.sin(np.arange(SR * 4))).astype(np.float32)
    assert find_cut(loud[: SR * 2], 3, 12) is None  # too short
    assert find_cut(np.concatenate([loud[: SR * 3], quiet[:SR]]), 3, 12) == SR * 4  # pause at the end
    assert find_cut(loud, 3, 12) is None  # still talking, under the maximum
    long = np.tile(loud, 4)
    long[SR * 8 : SR * 8 + FRAME] = 0  # one quiet frame after the minimum
    assert find_cut(long, 3, 12) == SR * 8 + FRAME


def test_live_transcriber_times_are_track_relative():
    """Segment times are offset by the audio consumed before their chunk."""
    eng = FakeEngine()
    got = []
    live = LiveTranscriber(eng, got.append, min_chunk_s=1, max_chunk_s=2)
    loud = (0.3 * np.sin(np.arange(SR * 5))).astype(np.float32)
    for i in range(0, len(loud), 1600):
        live.listener("system", loud[i : i + 1600])
    segs = live.stop()
    assert len(segs) == len(eng.calls)
    assert len(segs) >= 2
    assert segs[0].start == 0.5
    assert segs[1].start > 1.0  # second chunk is offset by the first
    assert all(s.speaker == "others" for s in segs)
    assert got == segs


def test_live_skips_silence():
    """Silent chunks are never sent to the engine."""
    eng = FakeEngine()
    live = LiveTranscriber(eng, lambda _s: None, min_chunk_s=1, max_chunk_s=2)
    live.listener("mic", np.zeros(SR * 3, np.float32))
    assert live.stop() == []
    assert eng.calls == []


def test_assign_speakers_by_overlap():
    """Each segment takes the speaker with the most overlap; segments without overlap keep theirs."""
    segs = [Segment(0, 2, "a"), Segment(2, 5, "b"), Segment(9, 10, "c", speaker="others")]
    assign_speakers(segs, [(0, 2.5, "speaker 1"), (2.5, 6, "speaker 2")])
    assert [s.speaker for s in segs] == ["speaker 1", "speaker 2", "others"]
