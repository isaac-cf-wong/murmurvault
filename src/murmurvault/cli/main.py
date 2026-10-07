"""Command-line interface. Every listing command accepts ``--json`` so agents can drive it."""

# Heavy modules (models, audio devices, the TUI) are imported inside the commands that need them, so
# `--help` and simple listing commands start instantly.
# ruff: noqa: PLC0415

from __future__ import annotations

import enum
import functools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from murmurvault import config as config_mod
from murmurvault.vault import Recording, Segment, Vault, fmt_time

app = typer.Typer(
    name="murmurvault",
    help="Local-first recording, transcription and semantic search.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()
err = Console(stderr=True)

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]
FolderOpt = Annotated[str, typer.Option("--folder", "-f", help="Folder path inside the vault, e.g. work/team.")]
TagsOpt = Annotated[list[str] | None, typer.Option("--tag", "-t", help="Tag (repeatable).")]
EngineOpt = Annotated[str | None, typer.Option("--engine", "-e", help="Engine name from the [engines] config.")]
ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Override the engine's model.")]


class LoggingLevel(enum.StrEnum):
    """Logging levels for the CLI."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(level: LoggingLevel = LoggingLevel.WARNING) -> None:
    """Send the package's log records to stderr through a Rich handler.

    Args:
        level: Logging level.
    """
    logger = logging.getLogger("murmurvault")
    logger.setLevel(level.value)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    handler = RichHandler(console=err, show_time=False, show_path=False, markup=False, level=level.value)
    logger.addHandler(handler)
    logger.propagate = False


@app.callback()
def main(
    verbose: Annotated[
        LoggingLevel, typer.Option("--verbose", "-v", help="Set verbosity level.")
    ] = LoggingLevel.WARNING,
) -> None:
    """Local-first recording, transcription and semantic search.

    Args:
        verbose: Verbosity level for logging.
    """
    setup_logging(verbose)


def _env() -> tuple[dict, Vault]:
    cfg = config_mod.load()
    return cfg, Vault(cfg["vault"])


def _emit(data: Any) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle_errors(fn):
    """Turn expected errors into a one-line message and exit code 1 instead of a traceback.

    Args:
        fn: The command function.

    Returns:
        The wrapped command.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            err.print(f"[red]error:[/red] {msg}")
            raise typer.Exit(1) from None

    return wrapper


def _rec_dict(vault: Vault, rec: Recording) -> dict:
    data = rec.to_dict()
    data["path"] = str(rec.path)
    data["transcripts"] = vault.transcript_names(rec)
    return data


def _progress(msg: str) -> None:
    err.print(f"[dim]… {msg}[/dim]")


# -- capture ------------------------------------------------------------------------------------


@app.command()
@handle_errors
def record(  # noqa: PLR0913, PLR0917 -- one parameter per command-line option
    title: Annotated[str, typer.Option("--title", "-T")] = "",
    folder: FolderOpt = "",
    tag: TagsOpt = None,
    mic: Annotated[bool, typer.Option("--mic/--no-mic", help="Record the microphone.")] = True,
    system: Annotated[bool, typer.Option("--system/--no-system", help="Record the computer's audio output.")] = True,
    live: Annotated[bool, typer.Option("--live/--no-live", help="Show a draft transcript while recording.")] = True,
    duration: Annotated[float | None, typer.Option("--duration", "-d", help="Stop after N seconds.")] = None,
    transcribe: Annotated[bool, typer.Option("--transcribe/--no-transcribe", help="Accurate pass after.")] = True,
    engine: EngineOpt = None,
    model: ModelOpt = None,
    as_json: JsonOpt = False,
):
    """Record the microphone and/or system audio as separate tracks. Stop with Ctrl-C."""
    from murmurvault.pipeline import transcribe_recording
    from murmurvault.session import RecordingSession

    cfg, vault = _env()

    def on_segment(seg: Segment) -> None:
        if as_json:
            _emit({"type": "segment", **seg.__dict__})
        else:
            console.print(f"[dim]{fmt_time(seg.start)}[/dim] [bold]{seg.speaker or ''}[/bold] {seg.text}")

    sess = RecordingSession(
        vault, cfg, title=title, folder=folder, tags=tag, mic=mic, system=system, live=live, on_segment=on_segment
    )
    sess.start()
    if not as_json:
        tracks = " + ".join(sess.recorder.sources)
        err.print(f"[green]● recording[/green] {sess.rec.id} ({tracks}) — Ctrl-C to stop")
    try:
        while duration is None or sess.elapsed() < duration:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    if not as_json:
        err.print("[yellow]■ stopping…[/yellow]")
    rec = sess.stop()
    if transcribe:
        transcribe_recording(vault, rec, cfg, engine=engine, model=model, progress=_progress)
    if as_json:
        _emit({"type": "recording", **_rec_dict(vault, rec)})
    else:
        err.print(f"saved {rec.id} ({fmt_time(rec.duration)}) → {rec.path}")


@app.command("import")
@handle_errors
def import_(  # noqa: PLR0913, PLR0917 -- one parameter per command-line option
    files: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False, help="Audio or video files.")],
    title: Annotated[str, typer.Option("--title", "-T")] = "",
    folder: FolderOpt = "",
    tag: TagsOpt = None,
    transcribe: Annotated[bool, typer.Option("--transcribe/--no-transcribe")] = True,
    engine: EngineOpt = None,
    model: ModelOpt = None,
    diarize: Annotated[bool | None, typer.Option("--diarize/--no-diarize")] = None,
    as_json: JsonOpt = False,
):
    """Import existing audio/video files (record first, transcribe later)."""
    from murmurvault.pipeline import import_file, transcribe_recording

    cfg, vault = _env()
    out = []
    for f in files:
        rec = import_file(vault, f, title=title if len(files) == 1 else "", folder=folder, tags=tag)
        if transcribe:
            transcribe_recording(vault, rec, cfg, engine=engine, model=model, diarization=diarize, progress=_progress)
        out.append(_rec_dict(vault, rec))
        if not as_json:
            err.print(f"imported {f.name} → {rec.id}")
    if as_json:
        _emit(out)


@app.command()
@handle_errors
def transcribe(  # noqa: PLR0913, PLR0917 -- one parameter per command-line option
    ref: Annotated[str, typer.Argument(help="Recording id or unique prefix.")],
    engine: EngineOpt = None,
    model: ModelOpt = None,
    language: Annotated[str | None, typer.Option("--language", "-l")] = None,
    diarize: Annotated[bool | None, typer.Option("--diarize/--no-diarize")] = None,
    as_json: JsonOpt = False,
):
    """(Re-)transcribe a recording. Each engine/model pass is kept; the newest becomes active."""
    from murmurvault.pipeline import transcribe_recording

    cfg, vault = _env()
    rec = vault.get(ref)
    tr = transcribe_recording(
        vault, rec, cfg, engine=engine, model=model, language=language, diarization=diarize, progress=_progress
    )
    if as_json:
        _emit(tr.to_dict())
    else:
        console.print(tr.text())


# -- browsing -----------------------------------------------------------------------------------


@app.command("ls")
@handle_errors
def ls(folder: FolderOpt = "", tag: TagsOpt = None, as_json: JsonOpt = False):
    """List recordings, newest first."""
    _, vault = _env()
    recs = vault.recordings(folder=folder or None, tags=tag)
    if as_json:
        _emit([_rec_dict(vault, r) for r in recs])
        return
    table = Table(box=None)
    table.add_column("id", no_wrap=True, min_width=20)
    for col in ("title", "folder", "tags", "length", "transcript"):
        table.add_column(col)
    for r in recs:
        table.add_row(r.id, r.title, r.folder, ",".join(r.tags), fmt_time(r.duration), r.active_transcript or "-")
    console.print(table)


@app.command()
@handle_errors
def show(
    ref: str,
    fmt: Annotated[str, typer.Option("--format", "-F", help="text | srt | json | meta")] = "text",
    transcript: Annotated[str | None, typer.Option("--transcript", help="Which pass (default: active).")] = None,
    timestamps: Annotated[bool, typer.Option("--timestamps/--no-timestamps")] = True,
):
    """Print a recording's transcript or metadata."""
    from murmurvault.pipeline import to_srt

    _, vault = _env()
    rec = vault.get(ref)
    if fmt == "meta":
        _emit(_rec_dict(vault, rec))
        return
    tr = vault.load_transcript(rec, transcript)
    if tr is None:
        raise RuntimeError(f"{rec.id} has no transcript yet; run `murmurvault transcribe {rec.id}`")
    if fmt == "json":
        _emit(tr.to_dict())
    elif fmt == "srt":
        sys.stdout.write(to_srt(tr))
    elif fmt == "text":
        sys.stdout.write(tr.text(timestamps=timestamps) + "\n")
    else:
        raise ValueError(f"unknown format {fmt!r}")


@app.command()
@handle_errors
def path(ref: str):
    """Print the directory holding a recording's files."""
    _, vault = _env()
    sys.stdout.write(f"{vault.get(ref).path}\n")


@app.command()
@handle_errors
def folders(as_json: JsonOpt = False):
    """List folders in use."""
    _, vault = _env()
    out = vault.folders()
    _emit(out) if as_json else console.print("\n".join(out) or "(none)")


@app.command()
@handle_errors
def tags(as_json: JsonOpt = False):
    """List tags with their recording counts."""
    _, vault = _env()
    out = vault.tags()
    if as_json:
        _emit(out)
    else:
        console.print("\n".join(f"{t}\t{n}" for t, n in out.items()) or "(none)")


# -- organising ---------------------------------------------------------------------------------


@app.command()
@handle_errors
def mv(ref: str, folder: Annotated[str, typer.Argument(help="Destination folder; '' for the vault root.")]):
    """Move a recording to another folder."""
    _, vault = _env()
    rec = vault.move(vault.get(ref), folder)
    err.print(f"{rec.id} → {rec.folder or '(root)'}")


@app.command()
@handle_errors
def tag(
    ref: str,
    names: Annotated[list[str], typer.Argument(help="Tags to add.")],
    remove: Annotated[bool, typer.Option("--remove", "-r", help="Remove the tags instead.")] = False,
):
    """Add (or with --remove, remove) tags."""
    _, vault = _env()
    rec = vault.get(ref)
    rec = vault.set_tags(rec, remove=names) if remove else vault.set_tags(rec, add=names)
    err.print(f"{rec.id}: {', '.join(rec.tags) or '(no tags)'}")


@app.command()
@handle_errors
def rename(ref: str, title: str):
    """Change a recording's title."""
    _, vault = _env()
    rec = vault.get(ref)
    rec.title = title
    vault.save(rec)


@app.command()
@handle_errors
def rm(ref: str, yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False):
    """Delete a recording and its files permanently."""
    from murmurvault.pipeline import delete_recording

    cfg, vault = _env()
    rec = vault.get(ref)
    if not yes and not typer.confirm(f"Delete {rec.id} ({rec.title}) and all its files?"):
        raise typer.Exit(1)
    delete_recording(vault, rec, cfg)


# -- search -------------------------------------------------------------------------------------


@app.command()
@handle_errors
def search(  # noqa: PLR0913, PLR0917 -- one parameter per command-line option
    query: str,
    k: Annotated[int, typer.Option("--limit", "-k")] = 10,
    folder: FolderOpt = "",
    tag: TagsOpt = None,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    as_json: JsonOpt = False,
):
    """Hybrid keyword + semantic search over all transcripts, then rerank."""
    from murmurvault.search import Index

    cfg, vault = _env()
    index = Index(vault, cfg)
    try:
        hits = index.search(query, k=k, folder=folder or None, tags=tag, rerank=rerank)
    finally:
        index.close()
    if as_json:
        _emit([h.to_dict() for h in hits])
        return
    for h in hits:
        console.print(
            f"[bold]{h.rec_id}[/bold] [cyan]{fmt_time(h.start)}[/cyan] {h.title} [dim]{h.folder} {h.score:.3f}[/dim]"
        )
        console.print(f"  {h.text[:300]}", highlight=False)
    if not hits:
        err.print("no matches")


@app.command()
@handle_errors
def reindex():
    """Rebuild the search index from the vault (e.g. after changing the embedding model)."""
    from murmurvault.search import Index

    cfg, vault = _env()
    index = Index(vault, cfg)
    try:
        n_recs, n_chunks = index.rebuild()
    finally:
        index.close()
    err.print(f"indexed {n_chunks} chunks from {n_recs} recordings")


# -- setup --------------------------------------------------------------------------------------


@app.command()
@handle_errors
def devices(as_json: JsonOpt = False):
    """List audio devices (for audio.mic_device / audio.system_device)."""
    from murmurvault.audio import list_devices

    devs = list_devices()
    if as_json:
        _emit(devs)
        return
    for d in devs:
        kind = ("in " if d["inputs"] else "   ") + ("out" if d["outputs"] else "")
        console.print(f"{d['index']:>3}  {kind}  {d['name']}")


@app.command()
@handle_errors
def config(init: Annotated[bool, typer.Option("--init", help="Write a commented config template.")] = False):
    """Show the config file location and the effective configuration."""
    path = config_mod.config_path()
    if init:
        if path.exists():
            raise RuntimeError(f"{path} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config_mod.TEMPLATE)
        err.print(f"wrote {path}")
        return
    err.print(f"config file: {path} ({'exists' if path.exists() else 'not created; using defaults'})")
    _emit(config_mod.load())


@app.command()
def tui():
    """Open the terminal UI."""
    from murmurvault.tui import run

    run()


if __name__ == "__main__":
    app()
