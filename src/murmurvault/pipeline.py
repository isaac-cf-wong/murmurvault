"""High-level operations shared by the CLI and the TUI."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from murmurvault import audio
from murmurvault.diarize import assign_speakers, diarize
from murmurvault.engines import get_engine
from murmurvault.search import Index
from murmurvault.vault import Recording, Segment, Transcript, Vault

# Speaker label per track when diarization does not assign one.
TRACK_SPEAKER = {"mic": "me", "system": "others"}

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


def import_file(vault: Vault, src: Path, title: str = "", folder: str = "", tags: list[str] | None = None) -> Recording:
    """Import an audio or video file as a new recording.

    The original file is kept as ``original.<ext>`` next to a 16 kHz FLAC track used for processing.

    Args:
        vault: The vault.
        src: File to import.
        title: Title; defaults to the file name.
        folder: Destination folder.
        tags: Tags to apply.

    Returns:
        The new recording, not yet transcribed.
    """
    data = audio.load_audio(src)
    rec = vault.new_recording(title=title or src.stem, folder=folder, tags=tags, source="import")
    try:
        shutil.copy2(src, rec.path / f"original{src.suffix.lower()}")
        audio.save_flac(rec.path / "audio.flac", data)
    except Exception:
        vault.delete(rec)
        raise
    rec.tracks = {"audio": "audio.flac"}
    rec.track_offsets = {"audio": 0.0}
    rec.duration = round(len(data) / audio.SR, 2)
    vault.save(rec)
    return rec


def transcribe_recording(  # noqa: PLR0913
    vault: Vault,
    rec: Recording,
    cfg: dict,
    *,
    engine: str | None = None,
    model: str | None = None,
    language: str | None = None,
    diarization: bool | None = None,
    progress: Progress = _noop,
) -> Transcript:
    """Run an accurate transcription pass over every track, save it as the active transcript and index it.

    Args:
        vault: The vault.
        rec: The recording.
        cfg: The configuration.
        engine: Engine name; defaults to ``transcription.engine``.
        model: Override the engine's model.
        language: Language code; defaults to ``transcription.language``.
        diarization: Split non-microphone tracks by speaker; defaults to ``diarization.enabled``.
        progress: Called with short status messages.

    Returns:
        The transcript, with segments from all tracks merged in time order.
    """
    eng = get_engine(cfg, engine, model)
    language = language if language is not None else (cfg["transcription"].get("language") or None)
    diarization = cfg["diarization"]["enabled"] if diarization is None else diarization

    segments: list[Segment] = []
    for track, filename in rec.tracks.items():
        progress(f"transcribing {track} with {eng.name}:{eng.model}")
        data = audio.load_audio(rec.path / filename)
        segs = eng.transcribe(data, language=language)
        default = TRACK_SPEAKER.get(track)
        for s in segs:
            s.track, s.speaker = track, default
        # The microphone is one person; diarize only tracks that can hold several speakers.
        if diarization and track != "mic" and segs:
            progress(f"diarizing {track}")
            assign_speakers(segs, diarize(data, cfg["diarization"]))
        offset = rec.track_offsets.get(track, 0.0)
        for s in segs:
            s.start, s.end = round(s.start + offset, 3), round(s.end + offset, 3)
        segments.extend(segs)

    segments.sort(key=lambda s: (s.start, s.end))
    transcript = Transcript(engine=eng.name, model=eng.model, segments=segments, language=language)
    name = f"{eng.name}-{eng.model}".replace("/", "_")
    vault.save_transcript(rec, name, transcript)
    progress("indexing")
    reindex_recording(vault, rec, cfg, transcript)
    return transcript


def reindex_recording(vault: Vault, rec: Recording, cfg: dict, transcript: Transcript | None = None) -> int:
    """Update the search index for one recording.

    Args:
        vault: The vault.
        rec: The recording.
        cfg: The configuration.
        transcript: Transcript to index; defaults to the active one.

    Returns:
        Number of chunks indexed.
    """
    index = Index(vault, cfg)
    try:
        return index.update(rec, transcript)
    finally:
        index.close()


def delete_recording(vault: Vault, rec: Recording, cfg: dict) -> None:
    """Delete a recording's files and remove it from the index.

    Args:
        vault: The vault.
        rec: The recording.
        cfg: The configuration.
    """
    index = Index(vault, cfg)
    try:
        index.remove(rec.id)
    finally:
        index.close()
    vault.delete(rec)


def to_srt(transcript: Transcript) -> str:
    """Render a transcript as SubRip subtitles.

    Args:
        transcript: The transcript.

    Returns:
        SRT text.
    """

    def ts(t: float) -> str:
        ms = round(t * 1000)
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, s in enumerate(transcript.segments, 1):
        who = f"{s.speaker}: " if s.speaker else ""
        blocks.append(f"{i}\n{ts(s.start)} --> {ts(s.end)}\n{who}{s.text}\n")
    return "\n".join(blocks)
