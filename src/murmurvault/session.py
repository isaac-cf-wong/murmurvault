"""A recording session: capture to disk, optionally with a live draft transcript."""

from __future__ import annotations

from collections.abc import Callable

from murmurvault.audio import Recorder
from murmurvault.engines import Engine, get_engine
from murmurvault.live import LiveTranscriber
from murmurvault.pipeline import reindex_recording
from murmurvault.vault import Recording, Segment, Transcript, Vault


def live_engine(cfg: dict) -> Engine:
    """Build the engine for the live draft pass.

    Args:
        cfg: The configuration.

    Returns:
        The engine named by ``live.engine``, using ``live.model`` if it is the built-in engine.
    """
    name = cfg["live"].get("engine") or cfg["transcription"]["engine"]
    section = cfg["engines"].get(name, {})
    # live.model only makes sense for the built-in engine; API engines keep their own model name.
    model = cfg["live"].get("model") if section.get("type", "faster-whisper") == "faster-whisper" else None
    return get_engine(cfg, name, model)


class RecordingSession:
    """Records to a new vault entry, optionally with a live draft transcript."""

    def __init__(  # noqa: PLR0913
        self,
        vault: Vault,
        cfg: dict,
        *,
        title: str = "",
        folder: str = "",
        tags: list[str] | None = None,
        mic: bool = True,
        system: bool = True,
        live: bool = False,
        on_segment: Callable[[Segment], None] | None = None,
    ):
        """Create the recording entry and set up capture; nothing is recorded until :meth:`start`.

        Args:
            vault: The vault.
            cfg: The configuration.
            title: Recording title.
            folder: Destination folder.
            tags: Tags to apply.
            mic: Record the microphone.
            system: Record the computer's audio output.
            live: Transcribe a draft while recording.
            on_segment: Called with each live segment.
        """
        self.vault, self.cfg = vault, cfg
        self.rec = vault.new_recording(title=title, folder=folder, tags=tags)
        try:
            self.recorder = Recorder(self.rec.path, cfg["audio"], mic=mic, system=system)
            self.live: LiveTranscriber | None = None
            if live:
                self.live = LiveTranscriber(
                    live_engine(cfg),
                    on_segment or (lambda s: None),
                    language=cfg["transcription"].get("language") or None,
                    min_chunk_s=float(cfg["live"]["min_chunk_s"]),
                    max_chunk_s=float(cfg["live"]["max_chunk_s"]),
                )
                self.recorder.add_listener(self.live.listener)
        except Exception:
            vault.delete(self.rec)
            raise

    def start(self) -> None:
        """Start recording; on failure the empty recording entry is removed."""
        try:
            self.recorder.start()
        except Exception:
            if self.live:
                self.live.stop()
            self.vault.delete(self.rec)
            raise

    def elapsed(self) -> float:
        """Seconds since recording started.

        Returns:
            Elapsed time.
        """
        return self.recorder.elapsed()

    def stop(self) -> Recording:
        """Stop recording, save the tracks and the live draft, and index the draft.

        Returns:
            The saved recording.

        Raises:
            RuntimeError: If no audio was captured; the empty entry is removed.
        """
        tracks, offsets, length = self.recorder.stop()
        draft = self.live.stop() if self.live else []
        if not tracks:
            self.vault.delete(self.rec)
            raise RuntimeError("no audio was captured (check devices with `murmurvault devices`)")
        rec = self.rec
        rec.tracks, rec.track_offsets, rec.duration = tracks, offsets, round(length, 2)
        self.vault.save(rec)
        if self.live and draft:
            for s in draft:
                off = offsets.get(s.track or "", 0.0)
                s.start, s.end = round(s.start + off, 3), round(s.end + off, 3)
            draft.sort(key=lambda s: s.start)
            eng = self.live.engine
            self.vault.save_transcript(rec, "live", Transcript(engine=eng.name, model=eng.model, segments=draft))
            reindex_recording(self.vault, rec, self.cfg)
        return rec
