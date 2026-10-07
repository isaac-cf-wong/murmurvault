"""Configuration: built-in defaults deep-merged with a user TOML file."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path
from typing import Any

import platformdirs

APP = "murmurvault"

DEFAULTS: dict[str, Any] = {
    "vault": str(platformdirs.user_data_path(APP) / "vault"),
    "transcription": {
        # Name of an entry under [engines] used for the accurate (final) pass.
        "engine": "local",
        # Empty string means auto-detect.
        "language": "",
    },
    "live": {
        # Engine used for the draft pass while recording; a small model keeps up in real time.
        "engine": "local",
        "model": "base",
        "min_chunk_s": 3.0,
        "max_chunk_s": 12.0,
    },
    "engines": {
        "local": {"type": "faster-whisper", "model": "small", "device": "auto", "compute_type": "default"},
        "openai": {
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": "whisper-1",
        },
    },
    "embedding": {
        # "local" (sentence-transformers), "openai" (any /v1/embeddings server) or "none".
        "backend": "local",
        "model": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "rerank": {
        # "local" (sentence-transformers CrossEncoder), "api" (any /v1/rerank server) or "none".
        "backend": "local",
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "base_url": "",
        "api_key_env": "RERANK_API_KEY",
    },
    "diarization": {
        "enabled": False,
        "model": "pyannote/speaker-diarization-community-1",
        "token_env": "HF_TOKEN",
    },
    "audio": {
        # sounddevice device name or index; empty means the system default input.
        "mic_device": "",
        # Optional loopback input device (e.g. "BlackHole 2ch"); empty means the native backend
        # (PulseAudio/PipeWire monitor on Linux, Core Audio process tap on macOS).
        "system_device": "",
    },
}

TEMPLATE = """\
# murmurvault configuration. Every key is optional; omitted keys use built-in defaults.
# vault = "~/murmurvault"

[transcription]
engine = "local"        # an entry under [engines]
language = ""           # "" = auto-detect, or e.g. "en", "nl"

[live]
engine = "local"
model = "base"          # smaller model for the real-time draft pass

[engines.local]
type = "faster-whisper"
model = "small"         # tiny, base, small, medium, large-v3, distil-large-v3, ...
device = "auto"         # auto, cpu, cuda

[engines.openai]
type = "openai"         # any OpenAI-compatible /v1/audio/transcriptions server
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "whisper-1"
# chunk_s = 30          # max seconds per upload, cut at a pause; servers that return one segment per
#                       # upload (e.g. Qwen3-ASR) give timestamps only this fine

# A local server, e.g. speaches or whisper.cpp's server:
# [engines.lan]
# type = "openai"
# base_url = "http://localhost:8000/v1"
# api_key_env = ""
# model = "Systran/faster-whisper-large-v3"

[embedding]
backend = "local"       # local, openai, none
model = "BAAI/bge-small-en-v1.5"

[rerank]
backend = "local"       # local, api, none
model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[diarization]
enabled = false         # needs `pip install murmurvault[diarize]` and an HF token with access to the model
model = "pyannote/speaker-diarization-community-1"
token_env = "HF_TOKEN"

[audio]
mic_device = ""
system_device = ""
"""


def config_path() -> Path:
    """Locate the user config file.

    Returns:
        ``$MURMURVAULT_CONFIG`` if set, else ``config.toml`` in the platform config directory.
    """
    env = os.environ.get("MURMURVAULT_CONFIG")
    return Path(env).expanduser() if env else platformdirs.user_config_path(APP) / "config.toml"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load(path: Path | None = None) -> dict[str, Any]:
    """Load the effective configuration: built-in defaults deep-merged with the user file.

    ``$MURMURVAULT_VAULT`` overrides the vault location.

    Args:
        path: Config file to read; defaults to :func:`config_path`.

    Returns:
        The configuration.
    """
    path = path or config_path()
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        with path.open("rb") as fh:
            cfg = _merge(cfg, tomllib.load(fh))
    if env_vault := os.environ.get("MURMURVAULT_VAULT"):
        cfg["vault"] = env_vault
    cfg["vault"] = str(Path(cfg["vault"]).expanduser())
    return cfg


def api_key(section: dict[str, Any]) -> str | None:
    """Read the API key named by a section's ``api_key_env``.

    Args:
        section: A config section such as ``engines.openai``.

    Returns:
        The key, or None if no variable is configured or set.
    """
    env = section.get("api_key_env") or ""
    return os.environ.get(env) if env else None
