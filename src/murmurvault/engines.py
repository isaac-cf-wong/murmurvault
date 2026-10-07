"""Transcription engines. Every engine takes 16 kHz mono float32 audio and returns timed segments."""

# faster-whisper and httpx are imported on first use: the first is an optional extra, and neither should
# slow down CLI commands that never transcribe.
# ruff: noqa: PLC0415

from __future__ import annotations

import io
from http import HTTPStatus
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from murmurvault import config as config_mod
from murmurvault.audio import SR, split_at_pauses
from murmurvault.vault import Segment


class Engine(Protocol):
    """A speech-to-text engine."""

    name: str
    model: str

    def transcribe(self, audio: np.ndarray, language: str | None = None, fast: bool = False) -> list[Segment]:
        """Transcribe audio.

        Args:
            audio: 16 kHz mono float32 samples.
            language: Language code, or None to auto-detect.
            fast: Trade accuracy for speed, for the live draft.

        Returns:
            Segments with times relative to the start of ``audio``.
        """


_WHISPER_CACHE: dict[tuple[str, str, str], Any] = {}


class FasterWhisperEngine:
    """The built-in engine: faster-whisper (CTranslate2) running locally."""

    def __init__(self, name: str, model: str = "small", device: str = "auto", compute_type: str = "default"):
        """Configure the engine; the model loads on first use and is cached per process.

        Args:
            name: Engine name from the config.
            model: Whisper model name or path, e.g. ``small`` or ``large-v3``.
            device: ``auto``, ``cpu`` or ``cuda``.
            compute_type: CTranslate2 compute type.
        """
        self.name, self.model, self.device, self.compute_type = name, model, device, compute_type

    def _load(self):
        key = (self.model, self.device, self.compute_type)
        if key not in _WHISPER_CACHE:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("the built-in engine needs `pip install 'murmurvault[whisper]'`") from exc
            _WHISPER_CACHE[key] = WhisperModel(self.model, device=self.device, compute_type=self.compute_type)
        return _WHISPER_CACHE[key]

    def transcribe(self, audio: np.ndarray, language: str | None = None, fast: bool = False) -> list[Segment]:
        """Transcribe with voice-activity filtering.

        Args:
            audio: 16 kHz mono float32 samples.
            language: Language code, or None to auto-detect.
            fast: Greedy decoding without conditioning on previous text.

        Returns:
            Non-empty segments.
        """
        model = self._load()
        segments, _info = model.transcribe(
            audio,
            language=language or None,
            beam_size=1 if fast else 5,
            vad_filter=True,
            condition_on_previous_text=not fast,
        )
        return [Segment(start=s.start, end=s.end, text=s.text.strip()) for s in segments if s.text.strip()]


class OpenAICompatibleEngine:
    """Any server implementing ``POST /v1/audio/transcriptions``: OpenAI, Groq, speaches, whisper.cpp, ..."""

    # Default upload length in seconds. Some servers (e.g. Qwen3-ASR on oMLX) return one segment per
    # upload, so the upload length bounds the timestamp resolution of the transcript and its search hits.
    CHUNK_S = 30.0

    def __init__(self, name: str, base_url: str, model: str, api_key: str | None = None, chunk_s: float = CHUNK_S):
        """Configure the endpoint.

        Args:
            name: Engine name from the config.
            base_url: API base URL, e.g. ``https://api.openai.com/v1``.
            model: Model name as the server knows it.
            api_key: Bearer token, if the server needs one.
            chunk_s: Maximum upload length in seconds; uploads end at the quietest moment in their second
                half. Keep it under 10 minutes for hosted APIs' ~25 MB upload limit.
        """
        self.name, self.model = name, model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chunk_s = chunk_s

    def _post(self, audio: np.ndarray, language: str | None, response_format: str) -> dict:
        import httpx

        buf = io.BytesIO()
        sf.write(buf, audio, SR, format="FLAC", subtype="PCM_16")
        data = {"model": self.model, "response_format": response_format}
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = httpx.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files={"file": ("audio.flac", buf.getvalue(), "audio/flac")},
                timeout=900,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"cannot reach {self.base_url}: {exc}") from exc
        if resp.status_code >= HTTPStatus.BAD_REQUEST:
            raise RuntimeError(f"{resp.status_code} from {self.base_url}: {resp.text[:300]}")
        return resp.json()

    def _one(self, audio: np.ndarray, language: str | None, offset: float) -> list[Segment]:
        try:
            body = self._post(audio, language, "verbose_json")
        except RuntimeError as exc:
            # Some models (e.g. gpt-4o-transcribe) only support plain json, which has no timestamps.
            if "response_format" not in str(exc) and "verbose_json" not in str(exc):
                raise
            body = self._post(audio, language, "json")
        segs = body.get("segments") or []
        if segs:
            return [
                Segment(start=offset + float(s["start"]), end=offset + float(s["end"]), text=s["text"].strip())
                for s in segs
                if s.get("text", "").strip()
            ]
        text = (body.get("text") or "").strip()
        return [Segment(start=offset, end=offset + len(audio) / SR, text=text)] if text else []

    def transcribe(self, audio: np.ndarray, language: str | None = None, fast: bool = False) -> list[Segment]:
        """Upload the audio in chunks cut at pauses and collect the segments.

        Args:
            audio: 16 kHz mono float32 samples.
            language: Language code, or None to auto-detect.
            fast: Ignored; the server decides how to decode.

        Returns:
            Segments with times relative to the start of ``audio``.
        """
        out: list[Segment] = []
        for start, end in split_at_pauses(audio, self.chunk_s):
            if end - start > SR // 10:
                out.extend(self._one(audio[start:end], language, start / SR))
        return out


def get_engine(cfg: dict, name: str | None = None, model: str | None = None) -> Engine:
    """Build an engine from the ``[engines]`` config.

    Args:
        cfg: The configuration.
        name: Engine name; defaults to ``transcription.engine``.
        model: Override the engine's configured model.

    Returns:
        The engine.

    Raises:
        KeyError: If no engine has that name.
        ValueError: If the engine's ``type`` is unknown.
    """
    name = name or cfg["transcription"]["engine"]
    try:
        section = cfg["engines"][name]
    except KeyError as exc:
        raise KeyError(f"unknown engine {name!r}; configured: {', '.join(cfg['engines'])}") from exc
    kind = section.get("type", "faster-whisper")
    if kind == "faster-whisper":
        return FasterWhisperEngine(
            name,
            model=model or section.get("model", "small"),
            device=section.get("device", "auto"),
            compute_type=section.get("compute_type", "default"),
        )
    if kind == "openai":
        return OpenAICompatibleEngine(
            name,
            base_url=section["base_url"],
            model=model or section["model"],
            api_key=config_mod.api_key(section),
            chunk_s=float(section.get("chunk_s", OpenAICompatibleEngine.CHUNK_S)),
        )
    raise ValueError(f"engine {name!r} has unknown type {kind!r} (expected faster-whisper or openai)")
