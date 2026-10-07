"""Terminal UI: browse folders, read transcripts, search, and record with a live draft."""

# Search, recording and transcription pull in models and audio devices; import them only when used.
# ruff: noqa: PLC0415

from __future__ import annotations

from typing import ClassVar

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, OptionList, RichLog, Static, Tree
from textual.widgets.option_list import Option

from murmurvault import config as config_mod
from murmurvault.vault import Recording, Segment, Vault, fmt_time


class Prompt(ModalScreen[str | None]):
    """A modal one-line text prompt."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, question: str, value: str = ""):
        """Create the prompt.

        Args:
            question: Label shown above the input.
            value: Initial input text.
        """
        super().__init__()
        self.question, self.value = question, value

    def compose(self) -> ComposeResult:
        """Lay out the prompt.

        Yields:
            The prompt widgets.
        """
        with Vertical(id="prompt"):
            yield Label(self.question)
            yield Input(value=self.value, id="prompt-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Return the entered text.

        Args:
            event: The submit event.
        """
        event.stop()
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        """Close without a value."""
        self.dismiss(None)


class MurmurApp(App):
    """Browse, search and record from the terminal."""

    TITLE = "murmurvault"
    CSS: ClassVar[str] = """
    #left { width: 38%; border-right: solid $panel; }
    #results { height: 35%; display: none; border-bottom: solid $panel; }
    #results.visible { display: block; }
    #status { height: 1; padding: 0 1; background: $panel; }
    Prompt { align: center middle; }
    #prompt { width: 70; height: auto; border: round $accent; padding: 1 2; background: $surface; }
    """
    BINDINGS: ClassVar = [
        ("slash", "focus_search", "Search"),
        ("r", "record", "Record/Stop"),
        ("t", "transcribe", "Transcribe"),
        ("g", "tag", "Tags"),
        ("m", "move", "Move"),
        ("e", "rename", "Rename"),
        ("f5", "refresh", "Refresh"),
        ("escape", "close_results", "Close results"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        """Load the configuration and open the vault."""
        super().__init__()
        self.cfg = config_mod.load()
        self.vault = Vault(self.cfg["vault"])
        self.selected: str | None = None
        self.hits: list = []
        self.session = None
        self.busy = ""
        # Kept as attributes: App.query_one only searches the active screen, which is a prompt while one is open.
        self.search_box = Input(placeholder="/ search transcripts (Enter)", id="search")
        self.folder_tree: Tree = Tree("vault", id="left")
        self.results_list = OptionList(id="results")
        self.transcript_view = RichLog(id="view", wrap=True, markup=True, highlight=False)
        self.status_bar = Static("", id="status")

    # -- layout -----------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        """Lay out the app.

        Yields:
            The app widgets.
        """
        yield Header()
        yield self.search_box
        with Horizontal():
            yield self.folder_tree
            with Vertical():
                yield self.results_list
                yield self.transcript_view
        yield self.status_bar
        yield Footer()

    def on_mount(self) -> None:
        """Fill the tree and start the status ticker."""
        self.refresh_tree()
        self.set_interval(0.5, self.update_status)
        self.folder_tree.focus()

    def update_status(self) -> None:
        """Refresh the status bar: recording time, live backlog and background work."""
        parts = [self.vault.root.as_posix()]
        if self.session is not None:
            parts.insert(0, f"● REC {fmt_time(self.session.elapsed())}")
            if self.session.live and self.session.live.backlog():
                parts.insert(1, f"live backlog {self.session.live.backlog()}")
        if self.busy:
            parts.insert(0, f"… {self.busy}")
        self.status_bar.update("  |  ".join(parts))

    def set_busy(self, msg: str) -> None:
        """Show what the background work is doing.

        Args:
            msg: Status message; empty to clear.
        """
        self.busy = msg

    # -- tree and transcript view -----------------------------------------------------------
    def refresh_tree(self) -> None:
        """Rebuild the folder tree from the vault."""
        tree = self.folder_tree
        tree.clear()
        tree.root.expand()
        nodes = {"": tree.root}

        def folder_node(path: str):
            if path not in nodes:
                parent, _, name = path.rpartition("/")
                nodes[path] = folder_node(parent).add(f"📁 {name}", data=None, expand=True)
            return nodes[path]

        for f in self.vault.folders():
            folder_node(f)
        for rec in sorted(self.vault.recordings(), key=lambda r: r.created, reverse=True):
            mark = "" if rec.active_transcript else " [dim](untranscribed)[/dim]"
            label = f"{escape(rec.title)} [dim]{fmt_time(rec.duration)}[/dim]{mark}"
            folder_node(rec.folder).add_leaf(label, data=rec.id)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Show the highlighted recording.

        Args:
            event: The highlight event.
        """
        if event.node.data:
            self.show_recording(event.node.data)

    def show_recording(self, rec_id: str, focus_time: float | None = None) -> None:
        """Show a recording's metadata and transcript.

        Args:
            rec_id: Recording id.
            focus_time: Highlight the segments around this time, e.g. a search hit.
        """
        try:
            rec = self.vault.get(rec_id)
        except KeyError:
            return
        self.selected = rec.id
        view = self.transcript_view
        view.clear()
        view.write(f"[b]{escape(rec.title)}[/b]  [dim]{rec.id}[/dim]")
        view.write(
            f"folder: {escape(rec.folder or '(root)')}   tags: {escape(', '.join(rec.tags) or '-')}   "
            f"length: {fmt_time(rec.duration)}   tracks: {', '.join(rec.tracks)}"
        )
        names = self.vault.transcript_names(rec)
        view.write(f"transcripts: {', '.join(names) or '-'}   active: {rec.active_transcript or '-'}\n")
        tr = self.vault.load_transcript(rec)
        if tr is None:
            view.write("[dim]No transcript yet — press t to transcribe.[/dim]")
            return
        for s in tr.segments:
            here = focus_time is not None and s.start <= focus_time + 0.5 and s.end >= focus_time - 0.5
            line = self._line(s)
            view.write(f"[reverse]{line}[/reverse]" if here else line)

    @staticmethod
    def _line(s: Segment) -> str:
        who = f"[b]{escape(s.speaker)}[/b] " if s.speaker else ""
        return f"[dim]{fmt_time(s.start)}[/dim] {who}{escape(s.text)}"

    # -- search -----------------------------------------------------------------------------
    def action_focus_search(self) -> None:
        """Focus the search box."""
        self.search_box.focus()

    def action_close_results(self) -> None:
        """Hide the search results."""
        self.results_list.remove_class("visible")
        self.folder_tree.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Run a search from the search box.

        Args:
            event: The submit event.
        """
        if event.input.id == "search" and event.value.strip():
            self.run_search(event.value.strip())

    @work(thread=True, exclusive=True, group="search")
    def run_search(self, query: str) -> None:
        """Search in a worker thread.

        Args:
            query: The query.
        """
        from murmurvault.search import Index

        self.call_from_thread(self.set_busy, f"searching {query!r}")
        try:
            index = Index(self.vault, self.cfg)
            try:
                hits = index.search(query, k=20)
            finally:
                index.close()
        except Exception as exc:  # noqa: BLE001 -- report in the UI instead of crashing it
            self.call_from_thread(self.notify, f"search failed: {exc}", severity="error")
            hits = []
        self.call_from_thread(self.show_hits, hits)

    def show_hits(self, hits: list) -> None:
        """List search hits.

        Args:
            hits: Hits from the index.
        """
        self.busy = ""
        self.hits = hits
        results = self.results_list
        results.clear_options()
        if not hits:
            self.notify("no matches")
            return
        results.add_options(
            Option(f"{fmt_time(h.start)}  {escape(h.title)} — {escape(h.text[:120])}", id=str(i))
            for i, h in enumerate(hits)
        )
        results.add_class("visible")
        results.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the recording of the selected hit at the hit's time.

        Args:
            event: The selection event.
        """
        hit = self.hits[int(event.option.id)]
        self.show_recording(hit.rec_id, focus_time=hit.start)

    # -- recording --------------------------------------------------------------------------
    def action_record(self) -> None:
        """Start or stop recording."""
        if self.session is None:
            self.start_recording()
        else:
            self.stop_recording()

    @work(thread=True, exclusive=True, group="record")
    def start_recording(self) -> None:
        """Start a live recording, falling back to the microphone alone if system audio is unavailable."""
        from murmurvault.session import RecordingSession

        self.call_from_thread(self.set_busy, "starting recorder (loading live model)")
        view = self.transcript_view

        def on_segment(seg: Segment) -> None:
            self.call_from_thread(view.write, self._line(seg))

        sess = None
        for system in (True, False):
            try:
                sess = RecordingSession(self.vault, self.cfg, live=True, system=system, on_segment=on_segment)
                sess.start()
                break
            except Exception as exc:  # noqa: BLE001 -- report in the UI instead of crashing it
                sess = None
                if not system:
                    self.call_from_thread(self.notify, f"cannot record: {exc}", severity="error")
                else:
                    msg = f"system audio unavailable ({exc}); mic only"
                    self.call_from_thread(self.notify, msg, severity="warning")
        self.call_from_thread(self.set_busy, "")
        if sess is None:
            return
        self.session = sess
        self.call_from_thread(view.clear)
        self.call_from_thread(view.write, f"[b red]● Recording[/b red] {sess.rec.id} — press r to stop\n")

    @work(thread=True, exclusive=True, group="record")
    def stop_recording(self) -> None:
        """Stop recording, then run the accurate pass."""
        sess, self.session = self.session, None
        if sess is None:
            return
        self.call_from_thread(self.set_busy, "finishing live draft")
        try:
            rec = sess.stop()
        except Exception as exc:  # noqa: BLE001 -- report in the UI instead of crashing it
            self.call_from_thread(self.notify, f"recording failed: {exc}", severity="error")
            self.call_from_thread(self.set_busy, "")
            return
        self.call_from_thread(self.refresh_tree)
        self._transcribe(rec)

    # -- editing ----------------------------------------------------------------------------
    def _current(self) -> Recording | None:
        if not self.selected:
            self.notify("select a recording first", severity="warning")
            return None
        try:
            return self.vault.get(self.selected)
        except KeyError:
            return None

    def action_transcribe(self) -> None:
        """Transcribe the selected recording."""
        if rec := self._current():
            self.transcribe_worker(rec)

    @work(thread=True, exclusive=True, group="transcribe")
    def transcribe_worker(self, rec: Recording) -> None:
        """Transcribe in a worker thread.

        Args:
            rec: The recording.
        """
        self._transcribe(rec)

    def _transcribe(self, rec: Recording) -> None:
        from murmurvault.pipeline import transcribe_recording

        try:
            transcribe_recording(self.vault, rec, self.cfg, progress=lambda m: self.call_from_thread(self.set_busy, m))
        except Exception as exc:  # noqa: BLE001 -- report in the UI instead of crashing it
            self.call_from_thread(self.notify, f"transcription failed: {exc}", severity="error")
        self.call_from_thread(self.set_busy, "")
        self.call_from_thread(self.refresh_tree)
        self.call_from_thread(self.show_recording, rec.id)

    def action_tag(self) -> None:
        """Edit the selected recording's tags."""
        rec = self._current()
        if not rec:
            return

        def done(value: str | None) -> None:
            if value is None:
                return
            try:
                self.vault.set_tags(rec, remove=rec.tags)
                self.vault.set_tags(rec, add=value.replace(",", " ").split())
            except ValueError as exc:
                self.notify(str(exc), severity="error")
            self.show_recording(rec.id)

        self.push_screen(Prompt("Tags (space-separated):", " ".join(rec.tags)), done)

    def action_move(self) -> None:
        """Move the selected recording to another folder."""
        rec = self._current()
        if not rec:
            return

        def done(value: str | None) -> None:
            if value is None:
                return
            try:
                self.vault.move(rec, value)
            except ValueError as exc:
                self.notify(str(exc), severity="error")
            self.refresh_tree()
            self.show_recording(rec.id)

        self.push_screen(Prompt("Move to folder (e.g. work/standups; empty = root):", rec.folder), done)

    def action_rename(self) -> None:
        """Rename the selected recording."""
        rec = self._current()
        if not rec:
            return

        def done(value: str | None) -> None:
            if value:
                rec.title = value
                self.vault.save(rec)
                self.refresh_tree()
                self.show_recording(rec.id)

        self.push_screen(Prompt("Title:", rec.title), done)

    def action_refresh(self) -> None:
        """Re-read the vault."""
        self.refresh_tree()

    def action_quit(self) -> None:
        """Quit, unless a recording is running."""
        if self.session is not None:
            self.notify("stop the recording first (r)", severity="warning")
            return
        self.exit()


def run() -> None:
    """Run the terminal UI."""
    MurmurApp().run()
