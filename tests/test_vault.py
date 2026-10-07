"""Tests for the on-disk vault."""

from __future__ import annotations

import json

import pytest

from murmurvault.vault import Segment, Transcript, Vault, normalize_folder, normalize_tag


@pytest.fixture
def vault(tmp_path):
    """Provide an empty vault.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        The vault.
    """
    return Vault(tmp_path / "vault")


def test_normalize_folder_cleans_slashes():
    """Leading, trailing and repeated slashes and ``.`` components are dropped."""
    assert normalize_folder("/work//team/") == "work/team"
    assert normalize_folder("a/./b") == "a/b"
    assert not normalize_folder(" / ")


@pytest.mark.parametrize("bad", ["../x", "a/../b", ".hidden", "a/.git"])
def test_normalize_folder_rejects_escapes_and_hidden(bad):
    """Folders cannot leave the vault or collide with hidden directories such as the index."""
    with pytest.raises(ValueError, match="invalid folder component"):
        normalize_folder(bad)


@pytest.mark.parametrize("bad", ["", "#", "two words", "a,b"])
def test_normalize_tag_rejects_invalid(bad):
    """Tags are single words without commas."""
    with pytest.raises(ValueError, match="invalid tag"):
        normalize_tag(bad)


def test_vault_roundtrip_move_tag_delete(vault):
    """Recordings can be created, found by prefix, moved, tagged and deleted."""
    rec = vault.new_recording(title="Standup", folder="work", tags=["Team", "#daily"])
    assert rec.tags == ["daily", "team"]
    assert vault.get(rec.id[:15]).title == "Standup"

    vault.move(rec, "work/archive")
    assert (vault.root / "work/archive" / rec.id / "meta.json").exists()
    assert vault.get(rec.id).folder == "work/archive"
    assert vault.folders() == ["work/archive"]

    vault.set_tags(rec, add=["x"], remove=["team"])
    assert vault.get(rec.id).tags == ["daily", "x"]
    assert [r.id for r in vault.recordings(tags=["x"])] == [rec.id]
    assert vault.recordings(folder="work")
    assert not vault.recordings(folder="home")

    vault.delete(rec)
    assert vault.recordings() == []
    assert not (vault.root / "work").exists()  # empty folders are pruned


def test_get_rejects_unknown_and_ambiguous(vault):
    """An id prefix must identify exactly one recording."""
    vault.new_recording()
    vault.new_recording()
    with pytest.raises(KeyError, match="ambiguous"):
        vault.get("2")
    with pytest.raises(KeyError, match="no recording"):
        vault.get("nope")


def test_folder_follows_directory_moved_outside_tool(vault):
    """The directory, not ``meta.json``, decides which folder a recording is in."""
    rec = vault.new_recording(folder="a")
    (vault.root / "b").mkdir()
    rec.path.rename(vault.root / "b" / rec.id)
    assert vault.get(rec.id).folder == "b"


def test_transcript_passes_are_kept(vault):
    """Every transcription pass is stored; the latest becomes active."""
    rec = vault.new_recording()
    vault.save_transcript(rec, "live", Transcript("local", "base", [Segment(0, 1, "draft")]))
    vault.save_transcript(rec, "local-small", Transcript("local", "small", [Segment(0, 1, "final", "me")]))
    rec = vault.get(rec.id)
    assert vault.transcript_names(rec) == ["live", "local-small"]
    assert rec.active_transcript == "local-small"
    assert vault.load_transcript(rec).segments[0].text == "final"
    assert vault.load_transcript(rec, "live").segments[0].text == "draft"
    assert json.loads((rec.path / "meta.json").read_text())["active_transcript"] == "local-small"
