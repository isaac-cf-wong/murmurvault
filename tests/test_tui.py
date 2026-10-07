"""Headless test of the terminal UI."""

from __future__ import annotations

import asyncio

import pytest

from murmurvault.vault import Segment, Transcript, Vault


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Create a vault with one transcribed, indexed recording and point the app at it.

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
    rec = vault.new_recording(title="Planning", folder="work/team", tags=["q4"])
    vault.save_transcript(rec, "x", Transcript("e", "m", [Segment(1, 4, "we agreed on the roadmap", "me")]))
    from murmurvault import config
    from murmurvault.search import Index

    index = Index(vault, config.load())
    index.rebuild()
    index.close()
    return vault, rec


def test_tui_browse_search_tag_move(env):
    """Browse to a recording, search it, then tag and move it through the prompts."""
    from murmurvault.tui import MurmurApp

    vault, rec = env

    async def scenario():
        app = MurmurApp()
        async with app.run_test(size=(120, 40)) as pilot:
            tree = app.folder_tree
            leaves = []

            def walk(node):
                for child in node.children:
                    if child.data:
                        leaves.append(child)
                    walk(child)

            walk(tree.root)
            assert [n.data for n in leaves] == [rec.id]
            tree.move_cursor(leaves[0])
            await pilot.pause()
            assert app.selected == rec.id
            view_text = "\n".join(line.text for line in app.transcript_view.lines)
            assert "roadmap" in view_text

            await pilot.press("slash")
            for ch in "roadmap":
                await pilot.press(ch)
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.results_list.option_count == 1

            app.folder_tree.focus()
            await pilot.press("g")
            await pilot.pause()
            app.screen.query_one("#prompt-input").value = "q4 urgent"
            await pilot.press("enter")
            await pilot.pause()
            assert vault.get(rec.id).tags == ["q4", "urgent"]

            # An invalid tag is rejected without losing the existing tags.
            await pilot.press("g")
            await pilot.pause()
            app.screen.query_one("#prompt-input").value = "keep #"
            await pilot.press("enter")
            await pilot.pause()
            assert vault.get(rec.id).tags == ["q4", "urgent"]

            await pilot.press("m")
            await pilot.pause()
            prompt_input = app.screen.query_one("#prompt-input")
            prompt_input.value = "archive"
            await pilot.press("enter")
            await pilot.pause()
            assert vault.get(rec.id).folder == "archive"

    asyncio.run(scenario())


def test_record_and_quit_are_ignored_while_the_recorder_starts(env, monkeypatch):
    """A second `r` or a `q` during the slow recorder start-up must not start a second recorder or exit."""
    from murmurvault.tui import MurmurApp

    async def scenario():
        app = MurmurApp()
        starts = []
        monkeypatch.setattr(app, "start_recording", lambda: starts.append(1))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("r")
            await pilot.press("r")
            await pilot.press("q")
            await pilot.pause()
            assert starts == [1]
            assert app.is_running

    asyncio.run(scenario())
