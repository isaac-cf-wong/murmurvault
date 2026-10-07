"""Real-time draft transcription.

Audio is always written to disk first (see ``audio.Recorder``); this module only listens to the tracks,
cuts them into chunks at pauses, and transcribes each chunk on a single worker thread. The draft is
replaced by an accurate pass over the full recording after it stops.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np

from murmurvault.audio import SR
from murmurvault.engines import Engine
from murmurvault.pipeline import TRACK_SPEAKER
from murmurvault.vault import Segment

logger = logging.getLogger(__name__)

FRAME = SR // 10  # 100 ms analysis frames
SILENCE_RMS = 0.01  # a trailing frame quieter than this counts as a pause
EMPTY_RMS = 0.003  # a whole chunk quieter than this is not sent to the engine


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def find_cut(buf: np.ndarray, min_s: float, max_s: float) -> int | None:
    """Decide where to end the current chunk.

    Cut at the end when the buffer ends in a pause; once it exceeds ``max_s``, cut at the quietest frame.

    Args:
        buf: Buffered 16 kHz samples not yet transcribed.
        min_s: Minimum chunk length in seconds.
        max_s: Maximum chunk length in seconds.

    Returns:
        Sample index to cut at, or None to keep buffering.
    """
    n = len(buf)
    if n < min_s * SR:
        return None
    if _rms(buf[-3 * FRAME :]) < SILENCE_RMS:
        return n
    if n < max_s * SR:
        return None
    # Forced cut: the quietest 100 ms frame after the minimum length, so words are rarely split.
    lo = int(min_s * SR) // FRAME
    frames = buf[: (n // FRAME) * FRAME].reshape(-1, FRAME)
    energies = np.sqrt(np.mean(np.square(frames[lo:]), axis=1))
    if not len(energies):
        return n
    return (lo + int(np.argmin(energies)) + 1) * FRAME


class LiveTranscriber:
    """Transcribes tracks in chunks while they are being recorded."""

    def __init__(
        self,
        engine: Engine,
        on_segment: Callable[[Segment], None],
        *,
        language: str | None = None,
        min_chunk_s: float = 3.0,
        max_chunk_s: float = 12.0,
    ):
        """Start the worker thread.

        Args:
            engine: Engine for the draft pass.
            on_segment: Called from the worker thread with each new segment.
            language: Language code, or None to auto-detect.
            min_chunk_s: Minimum chunk length in seconds.
            max_chunk_s: Maximum chunk length in seconds.
        """
        self.engine = engine
        self.on_segment = on_segment
        self.language = language
        self.min_s, self.max_s = min_chunk_s, max_chunk_s
        self.buffers: dict[str, np.ndarray] = {}
        self.consumed: dict[str, int] = {}
        self.segments: list[Segment] = []
        self.lock = threading.Lock()
        self.jobs: queue.Queue[tuple[str, int, np.ndarray] | None] = queue.Queue()
        self.worker = threading.Thread(target=self._run, name="live-transcriber", daemon=True)
        self.worker.start()

    def backlog(self) -> int:
        """Number of chunks waiting for the engine.

        Returns:
            Queue length; growing means the engine is slower than real time.
        """
        return self.jobs.qsize()

    def listener(self, track: str, block: np.ndarray) -> None:
        """Receive a block from a track's writer thread and queue a chunk when one is complete.

        Args:
            track: Track name.
            block: 16 kHz mono samples.
        """
        with self.lock:
            buf = np.concatenate([self.buffers.get(track, np.zeros(0, np.float32)), block])
            cut = find_cut(buf, self.min_s, self.max_s)
            if cut is not None:
                start = self.consumed.get(track, 0)
                self.jobs.put((track, start, buf[:cut]))
                self.consumed[track] = start + cut
                buf = buf[cut:]
            self.buffers[track] = buf

    def _run(self) -> None:
        while (job := self.jobs.get()) is not None:
            track, start, chunk = job
            if _rms(chunk) < EMPTY_RMS:
                continue
            try:
                segs = self.engine.transcribe(chunk, language=self.language, fast=True)
            except Exception as exc:  # noqa: BLE001 -- the draft is best effort; the audio is safe on disk
                logger.warning("live transcription failed: %s", exc)
                continue
            offset = start / SR
            for s in segs:
                seg = Segment(
                    start=round(offset + s.start, 3),
                    end=round(offset + s.end, 3),
                    text=s.text,
                    speaker=TRACK_SPEAKER.get(track),
                    track=track,
                )
                self.segments.append(seg)
                try:
                    self.on_segment(seg)
                except Exception as exc:  # noqa: BLE001 -- a display error must not stop the worker
                    logger.warning("live segment callback failed: %s", exc)

    def stop(self) -> list[Segment]:
        """Flush what is still buffered and wait for the backlog.

        Returns:
            The draft segments sorted by time, relative to each track's start.
        """
        with self.lock:
            for track, buf in self.buffers.items():
                if len(buf) > SR // 2:
                    self.jobs.put((track, self.consumed.get(track, 0), buf))
            self.buffers.clear()
        self.jobs.put(None)
        self.worker.join()
        return sorted(self.segments, key=lambda s: s.start)
