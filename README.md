# murmurvault

[![Python CI](https://github.com/isaac-cf-wong/murmurvault/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-cf-wong/murmurvault/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/isaac-cf-wong/murmurvault/main.svg)](https://results.pre-commit.ci/latest/github/isaac-cf-wong/murmurvault/main)
[![Documentation Status](https://github.com/isaac-cf-wong/murmurvault/actions/workflows/documentation.yml/badge.svg)](https://isaac-cf-wong.github.io/murmurvault/)
[![codecov](https://codecov.io/gh/isaac-cf-wong/murmurvault/graph/badge.svg)](https://codecov.io/gh/isaac-cf-wong/murmurvault)
[![PyPI Version](https://img.shields.io/pypi/v/murmurvault)](https://pypi.org/project/murmurvault/)
[![Python Versions](https://img.shields.io/pypi/pyversions/murmurvault)](https://pypi.org/project/murmurvault/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

Local-first recording, transcription and semantic search for meetings and voice
notes.

> **Status: early prototype.** The on-disk layout and the CLI may still change.

- **Your files, on your disk.** Each recording is a directory of plain files:
  FLAC tracks, JSON transcripts and a `meta.json`. Folders are real directories.
  The search index is derived data and can always be rebuilt.
- **Microphone and computer audio on separate tracks.** Your words are labelled
  `me` and the other side `others` without any model. Optional diarization
  splits `others` into individual speakers.
- **Live or later.** Get a draft transcript while recording, then an accurate
  pass when you stop. You can also import existing files and transcribe them at
  any time, with any engine. Every pass is kept.
- **Any engine.** Built-in
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper), or any
  OpenAI-compatible `/v1/audio/transcriptions` server: a hosted API, or a local
  one such as speaches or whisper.cpp's server.
- **Hybrid semantic search.** SQLite FTS5 (BM25) keyword search and embedding
  search, fused with reciprocal rank fusion, then reranked with a cross-encoder.
  Embedders and rerankers can be local models or API endpoints.
- **Folders and tags** for organising recordings.
- **CLI for scripts and agents** (`--json` everywhere) and a **TUI** for people.
  A GUI is planned.

## Install

```bash
pip install "murmurvault[whisper,search]"          # built-in engine + semantic search
pip install "murmurvault[whisper,search,diarize]"  # + speaker diarization (pyannote)
pip install "murmurvault[all]"                      # everything, incl. macOS system-audio capture
```

System requirements:

- **Linux:** PortAudio (`sudo apt install libportaudio2`) for the microphone.
  For the computer's audio, `parec` (`pulseaudio-utils`) or `pw-record`
  (PipeWire). Either works on PipeWire systems.
- **macOS 14.2+:** computer-audio capture uses Core Audio process taps via the
  `macos` extra. Grant your terminal both the microphone and the system-audio
  recording permissions. Alternatively, point `audio.system_device` at a
  loopback device such as BlackHole.

Use headphones in calls. Otherwise the far side leaks into your microphone
track.

## Quick start

```bash
murmurvault record --title "Weekly sync" -f work/team -t sync   # Ctrl-C to stop
murmurvault import ~/Downloads/interview.m4a -f research -t interview
murmurvault ls
murmurvault show <id>                 # transcript; --format srt|json|meta
murmurvault search "what did we decide about the budget"
murmurvault tui
```

Recording consent: tell the people you record. In many places it is a legal
requirement.

## Commands

| Command                     | What it does                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `record`                    | Record mic and/or system audio (`--no-mic`, `--no-system`, `--duration`, `--no-live`) |
| `import FILE...`            | Import audio/video files, transcribing them unless `--no-transcribe`                  |
| `transcribe ID`             | (Re-)transcribe with `--engine`, `--model`, `--language`, `--diarize`                 |
| `ls`, `show`, `path`        | List, print, locate recordings (`-f FOLDER`, `-t TAG` filters)                        |
| `mv`, `tag`, `rename`, `rm` | Organise recordings                                                                   |
| `folders`, `tags`           | List folders and tags                                                                 |
| `search QUERY`              | Hybrid search (`-k`, `-f`, `-t`, `--no-rerank`)                                       |
| `reindex`                   | Rebuild the index, e.g. after changing the embedding model                            |
| `devices`                   | List audio devices                                                                    |
| `config [--init]`           | Show the effective config, or write a commented template                              |
| `tui`                       | Terminal UI                                                                           |

IDs can be abbreviated to any unique prefix. With `--json`, `record` streams one
JSON object per live segment, followed by the final recording.

## TUI keys

`/` search · `r` start/stop recording (live draft) · `t` transcribe · `g` tags ·
`m` move · `e` rename · `F5` refresh · `Esc` close results · `q` quit

## Configuration

`murmurvault config --init` writes a commented `config.toml` to the platform
config directory. Set `MURMURVAULT_CONFIG` to use a different file and
`MURMURVAULT_VAULT` to use a different vault. Example of a local transcription
server:

```toml
[transcription]
engine = "lan"

[engines.lan]
type = "openai"
base_url = "http://localhost:8000/v1"
api_key_env = ""
model = "Systran/faster-whisper-large-v3"
```

Diarization uses `pyannote/speaker-diarization-community-1`. Accept its
conditions on Hugging Face, then export `HF_TOKEN`. Enable it with
`[diarization] enabled = true`, or per run with `--diarize`.

## Vault layout

```text
<vault>/work/team/20261007-101500-a1b2/
  meta.json                   title, tags, tracks, track offsets, active transcript
  mic.flac  system.flac       16 kHz mono tracks
  transcript.live.json        draft made while recording
  transcript.local-small.json accurate pass (one file per engine/model)
<vault>/.murmurvault/index.db search index (derived; `murmurvault reindex`)
```

## Development

```bash
uv sync --all-extras --group dev
uv run prek install
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause
