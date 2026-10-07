"""Tests for resampling and track writing."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import soundfile as sf

from murmurvault.audio import SR, Resampler, Track, split_at_pauses


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


def test_split_at_pauses_covers_audio_and_cuts_in_silence():
    """Chunks tile the audio, never exceed the maximum, and end at the quiet stretch."""
    loud = (0.3 * np.sin(np.arange(SR * 50))).astype(np.float32)
    loud[SR * 22 : SR * 22 + SR // 5] = 0  # 200 ms pause at 22 s, inside the second half of a 30 s window
    ranges = split_at_pauses(loud, 30)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(loud)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(ranges))
    assert all(0 < e - s <= 30 * SR for s, e in ranges)
    assert SR * 22 < ranges[0][1] <= SR * 22 + SR // 5


@pytest.mark.parametrize("n", [0, 1, SR * 30])
def test_split_at_pauses_short_audio(n):
    """Audio no longer than the maximum is one chunk; empty audio is none."""
    assert split_at_pauses(np.zeros(n, np.float32), 30) == ([(0, n)] if n else [])


@pytest.mark.parametrize(("max_s", "min_s"), [(0, None), (-1, None), (10, 11), (10, -1)])
def test_split_at_pauses_rejects_bad_bounds(max_s, min_s):
    """Non-positive maxima and minima outside [0, max] are errors, not infinite loops."""
    with pytest.raises(ValueError, match="min_s"):
        split_at_pauses(np.zeros(SR * 60, np.float32), max_s, min_s)
