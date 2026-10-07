"""Tests for resampling and track writing."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from murmurvault.audio import SR, Resampler, Track


def test_resampler_streaming_matches_length_and_tone():
    """Block-wise resampling keeps the length and phase of a continuous tone."""
    sr_in = 48000
    t = np.arange(sr_in * 2) / sr_in
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    r = Resampler(sr_in)
    out = np.concatenate([r(x[i : i + 1000]) for i in range(0, len(x), 1000)])
    assert abs(len(out) - 2 * SR) <= 2
    ref = np.sin(2 * np.pi * 440 * np.arange(len(out)) / SR)
    assert np.max(np.abs(out - ref)) < 0.02  # phase is continuous across blocks


def test_track_writes_flac_and_notifies(tmp_path):
    """Stereo blocks are downmixed, written as 16 kHz FLAC and passed to listeners."""
    seen = []
    track = Track("mic", tmp_path / "mic.flac", [lambda _name, b: seen.append(len(b))])
    for _ in range(10):
        track.feed(np.zeros((1600, 2), np.float32), SR)
    track.close()
    assert sf.info(str(tmp_path / "mic.flac")).frames == 16000
    assert sum(seen) == 16000


def test_track_survives_a_failing_listener(tmp_path):
    """A broken live consumer must not stop the audio from being written."""

    def broken(_name, _block):
        raise RuntimeError("boom")

    track = Track("mic", tmp_path / "mic.flac", [broken])
    track.feed(np.zeros(1600, np.float32), SR)
    track.close()
    assert sf.info(str(tmp_path / "mic.flac")).frames == 1600
