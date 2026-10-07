"""Tests for the command-line interface."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from murmurvault.cli.main import app
from murmurvault.vault import Segment, Transcript, Vault

runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolate the CLI with keyword-only search and one transcribed recording.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The vault and the recording.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text('[embedding]\nbackend = "none"\n[rerank]\nbackend = "none"\n')
    monkeypatch.setenv("MURMURVAULT_CONFIG", str(cfg))
    monkeypatch.setenv("MURMURVAULT_VAULT", str(tmp_path / "vault"))
    vault = Vault(tmp_path / "vault")
    rec = vault.new_recording(title="Retro", folder="work", tags=["team"])
    segs = [Segment(0, 2, "the deploy failed twice", "me"), Segment(2, 5, "we will add a canary", "others")]
    vault.save_transcript(rec, "x", Transcript("e", "m", segs))
    assert runner.invoke(app, ["reindex"]).exit_code == 0
    return vault, rec


def run_json(*args):
    """Run a command with ``--json`` and parse its output.

    Args:
        *args: Command-line arguments.

    Returns:
        The parsed JSON.
    """
    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_list_show_and_search(cli_env):
    """Listing, transcript output and search work and speak JSON."""
    _, rec = cli_env
    assert [r["id"] for r in run_json("ls")] == [rec.id]
    assert run_json("ls", "-t", "other") == []
    out = runner.invoke(app, ["show", rec.id[:15], "--no-timestamps"]).stdout
    assert out.splitlines() == ["me: the deploy failed twice", "others: we will add a canary"]
    srt = runner.invoke(app, ["show", rec.id, "-F", "srt"]).stdout
    assert "00:00:02,000 --> 00:00:05,000" in srt
    hits = run_json("search", "canary")
    assert [h["rec_id"] for h in hits] == [rec.id]
    assert hits[0]["start"] == 0


def test_organise(cli_env):
    """Tagging, moving and renaming update the vault."""
    vault, rec = cli_env
    assert runner.invoke(app, ["tag", rec.id, "Incident", "q4"]).exit_code == 0
    assert runner.invoke(app, ["tag", rec.id, "team", "--remove"]).exit_code == 0
    assert runner.invoke(app, ["mv", rec.id, "work/retros"]).exit_code == 0
    assert runner.invoke(app, ["rename", rec.id, "Deploy retro"]).exit_code == 0
    got = vault.get(rec.id)
    assert got.tags == ["incident", "q4"]
    assert got.folder == "work/retros"
    assert got.title == "Deploy retro"
    assert run_json("tags") == {"incident": 1, "q4": 1}
    assert run_json("folders") == ["work/retros"]


def test_import_without_transcription_and_delete(cli_env, tmp_path):
    """Imported files keep the original and get a 16 kHz track; rm deletes them."""
    vault, _ = cli_env
    wav = tmp_path / "memo.wav"
    sf.write(wav, np.zeros(44100, np.float32), 44100)
    [rec] = run_json("import", str(wav), "-f", "notes", "--no-transcribe")
    assert rec["tracks"] == {"audio": "audio.flac"}
    assert rec["duration"] == pytest.approx(1.0, abs=0.01)
    stored = vault.get(rec["id"])
    assert (stored.path / "original.wav").exists()
    assert runner.invoke(app, ["rm", rec["id"], "--yes"]).exit_code == 0
    assert run_json("ls", "-f", "notes") == []


def test_errors_are_one_line(cli_env):
    """Expected errors print a message and exit 1 instead of a traceback."""
    result = runner.invoke(app, ["show", "nope"])
    assert result.exit_code == 1
    assert "no recording matches" in result.output
    assert "Traceback" not in result.output


def test_error_text_keeps_square_brackets(cli_env, tmp_path, monkeypatch):
    """Error messages are printed literally, so an extra like ``pkg[av]`` is not eaten as markup."""
    from murmurvault import audio

    def fail(path):
        raise RuntimeError("install `murmurvault[av]` for PyAV support")

    monkeypatch.setattr(audio, "load_audio", fail)
    clip = tmp_path / "memo.m4a"
    clip.write_bytes(b"not audio")
    result = runner.invoke(app, ["import", str(clip), "--no-transcribe"])
    assert result.exit_code == 1
    assert "murmurvault[av]" in result.output


def test_import_continues_after_a_failed_file(cli_env, tmp_path):
    """One undecodable file is reported, the others are imported and listed, and the exit code is 1."""
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not audio")
    good = tmp_path / "memo.wav"
    sf.write(good, np.zeros(16000, np.float32), 16000)
    result = runner.invoke(app, ["import", str(bad), str(good), "--no-transcribe", "--json"])
    assert result.exit_code == 1
    assert "broken.wav" in result.output
    imported = json.loads(result.stdout)
    assert [r["title"] for r in imported] == ["memo"]


def test_import_reports_transcription_failure_with_id(cli_env, tmp_path, monkeypatch):
    """A file that imports but fails to transcribe is kept, listed, and named with its retry command."""
    from murmurvault import pipeline

    def fail(*args, **kwargs):
        raise RuntimeError("cannot reach http://h/v1: connection refused")

    monkeypatch.setattr(pipeline, "transcribe_recording", fail)
    good = tmp_path / "memo.wav"
    sf.write(good, np.zeros(16000, np.float32), 16000)
    result = runner.invoke(app, ["import", str(good), "--json"])
    assert result.exit_code == 1
    [rec] = json.loads(result.stdout)
    assert f"murmurvault transcribe {rec['id']}" in result.output.replace("\n", "")
