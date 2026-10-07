"""Optional speaker diarization ("who spoke when") with pyannote.audio."""

# torch and pyannote.audio come with the optional `diarize` extra and are imported on first use.
# ruff: noqa: PLC0415

from __future__ import annotations

import os
from typing import Any

import numpy as np

from murmurvault.audio import SR
from murmurvault.vault import Segment

Turn = tuple[float, float, str]
_PIPELINES: dict[str, Any] = {}


def _pipeline(cfg: dict):
    model = cfg["model"]
    if model in _PIPELINES:
        return _PIPELINES[model]
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("diarization needs `pip install 'murmurvault[diarize]'`") from exc
    token = os.environ.get(cfg.get("token_env") or "HF_TOKEN")
    pipe = Pipeline.from_pretrained(model, token=token)
    if pipe is None:
        raise RuntimeError(f"could not load {model}: accept its conditions on Hugging Face and set ${cfg['token_env']}")
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        pipe.to(torch.device("mps"))
    _PIPELINES[model] = pipe
    return pipe


def diarize(audio: np.ndarray, cfg: dict) -> list[Turn]:
    """Find speaker turns.

    Args:
        audio: 16 kHz mono float32 samples.
        cfg: The ``[diarization]`` config section.

    Returns:
        ``(start, end, label)`` turns, labelled ``speaker 1``, ``speaker 2``, ... in order of first appearance.
    """
    import torch

    out = _pipeline(cfg)({"waveform": torch.from_numpy(np.ascontiguousarray(audio))[None], "sample_rate": SR})
    annotation = getattr(out, "speaker_diarization", out)  # pyannote 4 wraps the annotation
    raw = [(float(t.start), float(t.end), str(label)) for t, _, label in annotation.itertracks(yield_label=True)]
    names: dict[str, str] = {}
    for _, _, label in sorted(raw):
        names.setdefault(label, f"speaker {len(names) + 1}")
    return [(s, e, names[label]) for s, e, label in raw]


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> None:
    """Label each segment with the speaker whose turns overlap it most; leave it unchanged if none do.

    Args:
        segments: Segments to label in place.
        turns: Speaker turns from :func:`diarize`.
    """
    for seg in segments:
        overlap: dict[str, float] = {}
        for start, end, label in turns:
            ov = min(seg.end, end) - max(seg.start, start)
            if ov > 0:
                overlap[label] = overlap.get(label, 0.0) + ov
        if overlap:
            seg.speaker = max(overlap, key=overlap.__getitem__)
