"""The vault: plain files on disk are the source of truth.

Layout::

    <vault>/
      <folder>/<sub-folder>/<recording-id>/
        meta.json                       title, tags, tracks, active transcript
        mic.flac, system.flac, ...      16 kHz mono tracks
        transcript.<name>.json          one file per transcription pass
      .murmurvault/index.db             derived search index; rebuild with `reindex`

Folders are real directories, so the vault can be browsed, backed up or grepped without the tool.
"""

from __future__ import annotations

import json
import secrets
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

META = "meta.json"
INTERNAL = ".murmurvault"


def _now() -> datetime:
    return datetime.now().astimezone()


@dataclass
class Segment:
    """A timed piece of transcript.

    Attributes:
        start: Start time in seconds from the start of the recording.
        end: End time in seconds.
        text: Transcribed text.
        speaker: Speaker label, e.g. ``me``, ``others`` or ``speaker 2``.
        track: Name of the track the segment came from.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None
    track: str | None = None


@dataclass
class Transcript:
    """One transcription pass over a recording.

    Attributes:
        engine: Name of the engine that produced it.
        model: Model used by the engine.
        segments: Segments sorted by start time.
        language: Language passed to the engine, if any.
        created: ISO timestamp of the pass.
    """

    engine: str
    model: str
    segments: list[Segment]
    language: str | None = None
    created: str = field(default_factory=lambda: _now().isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            The transcript as plain data.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Build a transcript from :meth:`to_dict` output.

        Args:
            data: Plain data as written to ``transcript.<name>.json``.

        Returns:
            The transcript.
        """
        segs = [Segment(**s) for s in data.get("segments", [])]
        return cls(
            engine=data["engine"],
            model=data["model"],
            segments=segs,
            language=data.get("language"),
            created=data.get("created", ""),
        )

    def text(self, timestamps: bool = True) -> str:
        """Render as plain text, one segment per line.

        Args:
            timestamps: Prefix each line with its start time.

        Returns:
            The rendered transcript.
        """
        lines = []
        for s in self.segments:
            who = f"{s.speaker}: " if s.speaker else ""
            stamp = f"[{fmt_time(s.start)}] " if timestamps else ""
            lines.append(f"{stamp}{who}{s.text.strip()}")
        return "\n".join(lines)


@dataclass
class Recording:
    """Metadata of one recording, stored as ``meta.json`` in its directory.

    Attributes:
        id: Unique id, also the directory name.
        title: Human-readable title.
        created: ISO timestamp of creation.
        folder: Folder path relative to the vault root; ``""`` is the root.
        tags: Sorted, normalised tags.
        tracks: Track name to file name inside the recording directory.
        track_offsets: Track name to start offset in seconds relative to the recording start.
        duration: Length in seconds.
        source: ``record`` or ``import``.
        active_transcript: Name of the transcript pass shown by default.
        path: Directory holding the recording's files (not serialised).
    """

    id: str
    title: str
    created: str
    folder: str = ""
    tags: list[str] = field(default_factory=list)
    tracks: dict[str, str] = field(default_factory=dict)
    track_offsets: dict[str, float] = field(default_factory=dict)
    duration: float = 0.0
    source: str = "record"
    active_transcript: str | None = None
    path: Path = field(default=Path(), compare=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON-compatible content of ``meta.json``.

        Returns:
            The metadata without the in-memory ``path``.
        """
        data = asdict(self)
        data.pop("path")
        return data


def fmt_time(seconds: float) -> str:
    """Format seconds as ``MM:SS``, or ``H:MM:SS`` from one hour on.

    Args:
        seconds: Time in seconds; negative values are clamped to zero.

    Returns:
        The formatted time.
    """
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def normalize_folder(folder: str) -> str:
    """Return a clean relative folder path, rejecting anything that escapes the vault.

    Args:
        folder: Folder path such as ``work/team``; leading and trailing slashes are ignored.

    Returns:
        The normalised path, ``""`` for the vault root.

    Raises:
        ValueError: If a component is ``..`` or starts with a dot.
    """
    folder = folder.strip().strip("/")
    if not folder:
        return ""
    parts = PurePosixPath(folder).parts
    for part in parts:
        if part in {"..", "."} or part.startswith("."):
            raise ValueError(f"invalid folder component: {part!r}")
    return "/".join(parts)


def normalize_tag(tag: str) -> str:
    """Normalise a tag: strip a leading ``#`` and lower-case it.

    Args:
        tag: The tag as typed.

    Returns:
        The normalised tag.

    Raises:
        ValueError: If the tag is empty or contains whitespace or commas.
    """
    tag = tag.strip().lstrip("#").lower()
    if not tag or any(c.isspace() or c == "," for c in tag):
        raise ValueError(f"invalid tag: {tag!r}")
    return tag


class Vault:
    """A directory tree of recordings."""

    def __init__(self, root: str | Path):
        """Open the vault at ``root``, creating it if needed.

        Args:
            root: Vault directory.
        """
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.internal = self.root / INTERNAL
        self.internal.mkdir(exist_ok=True)

    # -- recordings -------------------------------------------------------------------------
    def new_recording(
        self, title: str = "", folder: str = "", tags: list[str] | None = None, source: str = "record"
    ) -> Recording:
        """Create an empty recording directory with its ``meta.json``.

        Args:
            title: Title; defaults to one derived from the current time.
            folder: Folder path inside the vault.
            tags: Initial tags.
            source: ``record`` or ``import``.

        Returns:
            The new recording.
        """
        now = _now()
        rid = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        folder = normalize_folder(folder)
        path = self.root / folder / rid
        path.mkdir(parents=True)
        rec = Recording(
            id=rid,
            title=title or f"Recording {now:%Y-%m-%d %H:%M}",
            created=now.isoformat(timespec="seconds"),
            folder=folder,
            tags=sorted({normalize_tag(t) for t in tags or []}),
            source=source,
            path=path,
        )
        self.save(rec)
        return rec

    def save(self, rec: Recording) -> None:
        """Write ``meta.json`` atomically.

        Args:
            rec: The recording to save.
        """
        tmp = rec.path / (META + ".tmp")
        tmp.write_text(json.dumps(rec.to_dict(), indent=2) + "\n")
        tmp.replace(rec.path / META)

    def _load(self, meta: Path) -> Recording:
        data = json.loads(meta.read_text())
        rec = Recording(**data)
        rec.path = meta.parent
        # The directory is authoritative for the folder, so moves made outside the tool are honoured.
        rel = meta.parent.parent.relative_to(self.root)
        rec.folder = "" if str(rel) == "." else rel.as_posix()
        return rec

    def recordings(self, folder: str | None = None, tags: list[str] | None = None) -> list[Recording]:
        """List recordings, newest first.

        Args:
            folder: Only recordings in this folder or below it.
            tags: Only recordings carrying all of these tags.

        Returns:
            The matching recordings.
        """
        wanted_folder = normalize_folder(folder) if folder is not None else ""
        wanted_tags = {normalize_tag(t) for t in tags or []}
        out = []
        for meta in self.root.rglob(META):
            if INTERNAL in meta.relative_to(self.root).parts:
                continue
            try:
                rec = self._load(meta)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if wanted_folder and rec.folder != wanted_folder and not rec.folder.startswith(wanted_folder + "/"):
                continue
            if not wanted_tags <= set(rec.tags):
                continue
            out.append(rec)
        return sorted(out, key=lambda r: r.created, reverse=True)

    def get(self, ref: str) -> Recording:
        """Look up a recording by id or unique id prefix.

        Args:
            ref: Full id or a prefix of it.

        Returns:
            The recording.

        Raises:
            KeyError: If nothing matches or the prefix is ambiguous.
        """
        recs = self.recordings()
        matches = [r for r in recs if r.id == ref] or [r for r in recs if r.id.startswith(ref)]
        if not matches:
            raise KeyError(f"no recording matches {ref!r}")
        if len(matches) > 1:
            raise KeyError(f"{ref!r} is ambiguous: {', '.join(r.id for r in matches)}")
        return matches[0]

    def folders(self) -> list[str]:
        """List the folders that hold recordings.

        Returns:
            Sorted folder paths, excluding the root.
        """
        return sorted({r.folder for r in self.recordings()} - {""})

    def tags(self) -> dict[str, int]:
        """Count recordings per tag.

        Returns:
            Tag to number of recordings, sorted by tag.
        """
        counts: dict[str, int] = {}
        for r in self.recordings():
            for t in r.tags:
                counts[t] = counts.get(t, 0) + 1
        return dict(sorted(counts.items()))

    def move(self, rec: Recording, folder: str) -> Recording:
        """Move a recording to another folder, pruning folders left empty.

        Args:
            rec: The recording to move.
            folder: Destination folder; ``""`` for the root.

        Returns:
            The updated recording.
        """
        folder = normalize_folder(folder)
        dest = self.root / folder / rec.id
        if dest == rec.path:
            return rec
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rec.path), str(dest))
        self._prune_empty(rec.path.parent)
        rec.path, rec.folder = dest, folder
        return rec

    def delete(self, rec: Recording) -> None:
        """Delete a recording's directory and prune folders left empty.

        Args:
            rec: The recording to delete.
        """
        shutil.rmtree(rec.path)
        self._prune_empty(rec.path.parent)

    def _prune_empty(self, directory: Path) -> None:
        while directory != self.root and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
            directory = directory.parent

    def set_tags(self, rec: Recording, add: Iterable[str] = (), remove: Iterable[str] = ()) -> Recording:
        """Add and remove tags, then save.

        Args:
            rec: The recording to change.
            add: Tags to add.
            remove: Tags to remove.

        Returns:
            The updated recording.
        """
        tags = set(rec.tags) | {normalize_tag(t) for t in add}
        tags -= {normalize_tag(t) for t in remove}
        rec.tags = sorted(tags)
        self.save(rec)
        return rec

    # -- transcripts ------------------------------------------------------------------------
    @staticmethod
    def transcript_names(rec: Recording) -> list[str]:
        """List the transcription passes stored for a recording.

        Args:
            rec: The recording.

        Returns:
            Sorted pass names.
        """
        return sorted(p.name[len("transcript.") : -len(".json")] for p in rec.path.glob("transcript.*.json"))

    def save_transcript(self, rec: Recording, name: str, transcript: Transcript, activate: bool = True) -> None:
        """Write a transcription pass, optionally making it the active one.

        Args:
            rec: The recording.
            name: Pass name, used in the file name.
            transcript: The transcript to write.
            activate: Make it the transcript shown and indexed by default.
        """
        path = rec.path / f"transcript.{name}.json"
        path.write_text(json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False) + "\n")
        if activate:
            rec.active_transcript = name
            self.save(rec)

    @staticmethod
    def load_transcript(rec: Recording, name: str | None = None) -> Transcript | None:
        """Read a transcription pass.

        Args:
            rec: The recording.
            name: Pass name; defaults to the active one.

        Returns:
            The transcript, or None if there is none.
        """
        name = name or rec.active_transcript
        if not name:
            return None
        path = rec.path / f"transcript.{name}.json"
        if not path.exists():
            return None
        return Transcript.from_dict(json.loads(path.read_text()))
