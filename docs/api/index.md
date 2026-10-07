---
title: API Reference
description: Reference for the murmurvault Python modules.
icon: material/api
---

The command line (`murmurvault --help`) and the terminal UI are the main
interfaces. These pages document the modules underneath them for scripting and
development.

## Modules

- **[Vault](vault.md)** – Recordings, transcripts, folders and tags on disk.
- **[Audio](audio.md)** – Capture devices, tracks, resampling and decoding.
- **[Engines](engines.md)** – Transcription engines: built-in faster-whisper and
  OpenAI-compatible APIs.
- **[Live transcription](live.md)** – Real-time draft transcription while
  recording.
- **[Recording session](session.md)** – Capture plus live draft, as used by the
  CLI and TUI.
- **[Pipeline](pipeline.md)** – Import, transcription passes, indexing and
  export.
- **[Diarization](diarize.md)** – Optional speaker diarization with
  pyannote.audio.
- **[Search](search.md)** – Hybrid keyword and semantic search with reranking.
- **[Configuration](config.md)** – Defaults and the user configuration file.
