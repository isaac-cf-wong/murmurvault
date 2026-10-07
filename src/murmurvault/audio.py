"""Audio capture and file I/O.

Every recording keeps the microphone and the computer's own audio on separate 16 kHz mono tracks. Keeping
them apart gives a free "me vs. others" speaker split and gives diarization a cleaner signal.
"""

# Optional and platform-specific dependencies (sounddevice needs PortAudio, catap is macOS-only, PyAV
# comes with the av or whisper extra) are imported where they are used, so the rest of the package works
# without them.
# ruff: noqa: PLC0415

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Sample rate of every stored track, in Hz.
SR = 16000

Listener = Callable[[str, np.ndarray], None]
Feed = Callable[[np.ndarray, float], None]

_INT16_SCALE = 32768.0
_INT16_BYTES = 2
_FLOAT32_BYTES = 4
_MAX_CHANNELS = 2
_FRAMES_CHANNELS_NDIM = 2
_MIN_INTERP_SAMPLES = 2
_RATE_TOLERANCE = 1e-6


def to_mono(x: np.ndarray) -> np.ndarray:
    """Average the channels of a ``(frames, channels)`` block.

    Args:
        x: Mono or multi-channel samples.

    Returns:
        Mono float32 samples.
    """
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == _FRAMES_CHANNELS_NDIM else x


class Resampler:
    """Streaming linear-interpolation resampler that keeps phase across blocks."""

    def __init__(self, sr_in: float, sr_out: float = SR):
        """Create a resampler.

        Args:
            sr_in: Input sample rate in Hz.
            sr_out: Output sample rate in Hz.
        """
        self.step = sr_in / sr_out
        self.pos = 0.0
        self.buf = np.zeros(0, dtype=np.float32)
        self.identity = abs(sr_in - sr_out) < _RATE_TOLERANCE

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Resample the next block.

        Args:
            x: Mono input samples continuing the previous block.

        Returns:
            The output samples that the input so far determines.
        """
        if self.identity:
            return x
        self.buf = np.concatenate([self.buf, x])
        n = len(self.buf)
        if n < _MIN_INTERP_SAMPLES:
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(self.pos, n - 1, self.step)
        out = np.interp(idx, np.arange(n), self.buf).astype(np.float32)
        nxt = self.pos + len(idx) * self.step
        # The next position can lie past the buffer; keep it relative to the samples still to come.
        drop = min(int(nxt), n)
        self.buf = self.buf[drop:]
        self.pos = nxt - drop
        return out


def resample(x: np.ndarray, sr_in: float, sr_out: float = SR) -> np.ndarray:
    """Resample a whole signal to mono at ``sr_out``.

    Args:
        x: Mono or multi-channel samples.
        sr_in: Input sample rate in Hz.
        sr_out: Output sample rate in Hz.

    Returns:
        Mono float32 samples.
    """
    return Resampler(sr_in, sr_out)(to_mono(x))


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio or video file to 16 kHz mono float32.

    libsndfile handles WAV/FLAC/OGG/MP3; anything else (m4a, webm, mp4, ...) goes through PyAV, which the
    ``av`` extra installs (``whisper`` brings it too).

    Args:
        path: File to decode.

    Returns:
        The samples.

    Raises:
        RuntimeError: If the file cannot be decoded.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return resample(data, sr)
    except RuntimeError:  # soundfile.LibsndfileError: format not supported by libsndfile
        pass
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(f"cannot decode {path}: install `murmurvault[av]` for PyAV support") from exc
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise RuntimeError(f"{path} has no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in container.decode(container.streams.audio[0]):
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def split_at_pauses(audio: np.ndarray, max_s: float, min_s: float | None = None) -> list[tuple[int, int]]:
    """Cut a recording into chunks of at most ``max_s``, each ending at the quietest 100 ms frame.

    Args:
        audio: 16 kHz mono samples.
        max_s: Maximum chunk length in seconds; at least one 100 ms frame.
        min_s: Earliest point in a chunk to look for a pause; defaults to half of ``max_s``.

    Returns:
        ``(start, end)`` sample ranges covering ``audio`` without gaps.

    Raises:
        ValueError: If ``max_s`` is shorter than a frame or ``min_s`` is not in ``[0, max_s]``.
    """
    frame = SR // 10
    min_s = max_s / 2 if min_s is None else min_s
    if max_s * SR < frame or not 0 <= min_s <= max_s:
        raise ValueError(f"need 0 <= min_s <= max_s and max_s >= 0.1, got min_s={min_s}, max_s={max_s}")
    max_n = int(max_s * SR)
    min_n = min(int(min_s * SR), max_n - frame)
    ranges: list[tuple[int, int]] = []
    start = 0
    while len(audio) - start > max_n:
        lo = start + (min_n // frame) * frame
        hi = start + (max_n // frame) * frame
        frames = audio[lo:hi].reshape(-1, frame)
        cut = lo + (int(np.argmin(np.mean(np.square(frames), axis=1))) + 1) * frame
        ranges.append((start, cut))
        start = cut
    if start < len(audio):
        ranges.append((start, len(audio)))
    return ranges


def save_flac(path: Path, audio: np.ndarray) -> None:
    """Write 16 kHz mono samples as 16-bit FLAC.

    Args:
        path: Destination file.
        audio: Samples at :data:`SR`.
    """
    sf.write(str(path), audio, SR, format="FLAC", subtype="PCM_16")


# -- tracks -------------------------------------------------------------------------------------


class Track:
    """Receives raw blocks from a source thread, resamples to 16 kHz and writes FLAC off-thread."""

    def __init__(self, name: str, path: Path, listeners: list[Listener]):
        """Open the track file and start its writer thread.

        Args:
            name: Track name, e.g. ``mic`` or ``system``.
            path: FLAC file to write.
            listeners: Callbacks receiving each resampled block, e.g. the live transcriber.
        """
        self.name = name
        self.path = path
        self.listeners = listeners
        self.file = sf.SoundFile(str(path), "w", SR, 1, format="FLAC", subtype="PCM_16")
        self.queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self.frames = 0
        self.first_block_time: float | None = None
        self.resampler: Resampler | None = None
        self.thread = threading.Thread(target=self._run, name=f"track-{name}", daemon=True)
        self.thread.start()

    def feed(self, block: np.ndarray, sr: float) -> None:
        """Queue a block of samples. Safe to call from a real-time audio callback.

        Args:
            block: Mono or multi-channel samples.
            sr: Sample rate of the block in Hz.
        """
        if self.first_block_time is None:
            # Time at which the first sample of this block was captured.
            self.first_block_time = time.monotonic() - len(block) / sr
        self.queue.put((to_mono(block).copy(), sr))

    def _run(self) -> None:
        while (item := self.queue.get()) is not None:
            block, sr = item
            if self.resampler is None:
                self.resampler = Resampler(sr)
            out = self.resampler(block)
            if not len(out):
                continue
            self.file.write(out)
            self.frames += len(out)
            for listener in self.listeners:
                try:
                    listener(self.name, out)
                except Exception as exc:  # noqa: BLE001 -- a broken live consumer must never lose audio
                    logger.warning("live listener failed: %s", exc)

    def close(self) -> None:
        """Flush queued blocks and close the file."""
        self.queue.put(None)
        self.thread.join()
        self.file.close()


# -- sources ------------------------------------------------------------------------------------


class Source:
    """An audio input that pushes blocks to a feed callback until stopped."""

    def start(self, feed: Feed) -> None:
        """Start capturing.

        Args:
            feed: Called with ``(samples, sample_rate)`` for every captured block.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Stop capturing and release the device."""
        raise NotImplementedError


def _device(value: str | int | None) -> str | int | None:
    if value in {None, ""}:
        return None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


class SoundDeviceSource(Source):
    """Any PortAudio input: the microphone, or a loopback device such as BlackHole."""

    def __init__(self, device: str | int | None = None):
        """Select the input device.

        Args:
            device: Device name or index; None or ``""`` for the default input.
        """
        self.device = _device(device)
        self.stream: Any = None

    def start(self, feed: Feed) -> None:
        """Open the device at its native rate and start streaming.

        Args:
            feed: Called with every captured block.
        """
        import sounddevice as sd

        info = sd.query_devices(self.device, "input")
        sr = float(info["default_samplerate"])
        channels = max(1, min(_MAX_CHANNELS, int(info["max_input_channels"])))

        def callback(indata, _frames, _time_info, _status):
            feed(indata, sr)

        self.stream = sd.InputStream(
            device=self.device, channels=channels, samplerate=sr, dtype="float32", callback=callback
        )
        self.stream.start()

    def stop(self) -> None:
        """Close the stream."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class PulseMonitorSource(Source):
    """Linux: capture what the speakers play via the PulseAudio/PipeWire monitor of the default sink."""

    def __init__(self):
        """Pick ``parec`` or ``pw-record``.

        Raises:
            RuntimeError: If neither tool is installed.
        """
        if parec := shutil.which("parec"):
            self.cmd = [parec, "--device=@DEFAULT_MONITOR@", "--format=s16le", f"--rate={SR}", "--channels=1"]
        elif pw_record := shutil.which("pw-record"):
            self.cmd = [
                pw_record, "-P", "{ stream.capture.sink = true }",
                "--rate", str(SR), "--channels", "1", "--format", "s16", "-",
            ]  # fmt: skip
        else:
            raise RuntimeError(
                "system audio capture needs `parec` (pulseaudio-utils) or `pw-record` (pipewire); "
                "install one, set audio.system_device to a loopback input, or pass --no-system"
            )
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    def start(self, feed: Feed) -> None:
        """Start the capture process and a thread reading its raw PCM output.

        Args:
            feed: Called with every captured block.
        """
        # The command is a fixed argument list built in __init__; nothing user-supplied reaches it.
        proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # noqa: S603
        self.proc = proc
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError("capture process has no stdout")

        def reader():
            block = SR // 10 * _INT16_BYTES  # 100 ms of int16
            while chunk := stdout.read(block):
                usable = len(chunk) - len(chunk) % _INT16_BYTES
                feed(np.frombuffer(chunk[:usable], dtype=np.int16).astype(np.float32) / _INT16_SCALE, SR)

        self.thread = threading.Thread(target=reader, name="system-audio", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Terminate the capture process."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=3)


class CoreAudioTapSource(Source):
    """macOS 14.2+: capture the system output mix through a Core Audio process tap (via ``catap``)."""

    def __init__(self, scratch: Path):
        """Check that ``catap`` is installed.

        Args:
            scratch: Path for the WAV file catap writes alongside the stream; removed on stop.

        Raises:
            RuntimeError: If ``catap`` is missing.
        """
        try:
            import catap  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "system audio capture on macOS needs `pip install 'murmurvault[macos]'` (macOS 14.2+), "
                "or set audio.system_device to a loopback input such as BlackHole"
            ) from exc
        self.scratch = scratch
        self.session: Any = None

    def start(self, feed: Feed) -> None:
        """Create the tap and start streaming buffers.

        Args:
            feed: Called with every captured block.
        """
        from catap import record_system_audio

        def on_buffer(buf):
            raw = bytes(buf.bytes)
            channels = max(1, int(getattr(buf.format, "channels", 1)))
            width = len(raw) // max(1, buf.frame_count * channels)
            if width == _FLOAT32_BYTES:
                data = np.frombuffer(raw, dtype=np.float32)
            elif width == _INT16_BYTES:
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _INT16_SCALE
            else:
                return
            data = data[: (len(data) // channels) * channels].reshape(-1, channels)
            feed(data, float(buf.format.sample_rate))

        self.session = record_system_audio(output_path=str(self.scratch), mono=True, on_buffer=on_buffer)
        self.session.start()

    def stop(self) -> None:
        """Stop the tap and remove catap's own WAV file."""
        if self.session is not None:
            self.session.stop()
        self.scratch.unlink(missing_ok=True)


def system_source(audio_cfg: dict, rec_dir: Path) -> Source:
    """Choose how to capture the computer's audio output on this platform.

    Args:
        audio_cfg: The ``[audio]`` config section.
        rec_dir: Recording directory, for scratch files.

    Returns:
        The source.

    Raises:
        RuntimeError: If the platform is not supported.
    """
    if audio_cfg.get("system_device"):
        return SoundDeviceSource(audio_cfg["system_device"])
    if sys.platform.startswith("linux"):
        return PulseMonitorSource()
    if sys.platform == "darwin":
        return CoreAudioTapSource(rec_dir / ".catap-system.wav")
    raise RuntimeError(f"system audio capture is not supported on {sys.platform} yet; pass --no-system")


class Recorder:
    """Captures one or more sources, each to its own track file."""

    def __init__(self, rec_dir: Path, audio_cfg: dict, mic: bool = True, system: bool = True):
        """Set up the sources.

        Args:
            rec_dir: Recording directory to write the tracks into.
            audio_cfg: The ``[audio]`` config section.
            mic: Record the microphone.
            system: Record the computer's audio output.

        Raises:
            ValueError: If both sources are disabled.
        """
        if not (mic or system):
            raise ValueError("nothing to record: enable the microphone or system audio")
        self.rec_dir = rec_dir
        self.listeners: list[Listener] = []
        self.sources: dict[str, Source] = {}
        if mic:
            self.sources["mic"] = SoundDeviceSource(audio_cfg.get("mic_device"))
        if system:
            self.sources["system"] = system_source(audio_cfg, rec_dir)
        self.tracks: dict[str, Track] = {}
        self.t0 = 0.0

    def add_listener(self, fn: Listener) -> None:
        """Register a callback for every resampled block of every track.

        Args:
            fn: Called with ``(track_name, samples)``.
        """
        self.listeners.append(fn)

    def start(self) -> None:
        """Open the track files and start every source; on failure, stop what was started."""
        self.t0 = time.monotonic()
        try:
            for name, source in self.sources.items():
                track = Track(name, self.rec_dir / f"{name}.flac", self.listeners)
                self.tracks[name] = track
                source.start(track.feed)
        except Exception:
            self.stop()
            raise

    def elapsed(self) -> float:
        """Seconds since :meth:`start`.

        Returns:
            Elapsed time.
        """
        return time.monotonic() - self.t0

    def stop(self) -> tuple[dict[str, str], dict[str, float], float]:
        """Stop all sources and close the tracks; empty tracks are deleted.

        Returns:
            Track name to file name, track name to start offset in seconds, and the total length.
        """
        for source in self.sources.values():
            try:
                source.stop()
            except Exception as exc:  # noqa: BLE001 -- one failing device must not keep the others open
                logger.warning("stopping a source failed: %s", exc)
        tracks, offsets, longest = {}, {}, 0.0
        for name, track in self.tracks.items():
            track.close()
            if track.frames == 0:
                track.path.unlink(missing_ok=True)
                continue
            tracks[name] = track.path.name
            first = track.first_block_time if track.first_block_time is not None else self.t0
            offsets[name] = round(max(0.0, first - self.t0), 3)
            longest = max(longest, offsets[name] + track.frames / SR)
        return tracks, offsets, longest


def list_devices() -> list[dict[str, Any]]:
    """List PortAudio devices.

    Returns:
        One dict per device with its index, name, channel counts and default rate.
    """
    import sounddevice as sd

    return [
        {
            "index": idx,
            "name": dev["name"],
            "inputs": dev["max_input_channels"],
            "outputs": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        }
        for idx, dev in enumerate(sd.query_devices())
    ]
