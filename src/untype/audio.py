"""Audio capture for whispered-speech input.

Provides push-to-talk style recording via sounddevice. All audio stays
in memory -- nothing ever touches disk.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Push-to-talk microphone recorder.

    Parameters
    ----------
    sample_rate:
        Samples per second.  16 kHz is the default expected by most
        speech-to-text models.
    device:
        PortAudio device index or name.  ``None`` uses the system default
        input device.
    on_volume:
        Optional callback invoked with current audio level (0.0-1.0)
        during recording. Called from the audio thread, so must be
        thread-safe.
    on_audio_chunk:
        Optional callback invoked with each audio chunk during recording.
        Called from the audio thread with Float32 numpy array.
        Allows realtime streaming to STT engines.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: int | str | None = None,
        on_volume: Callable[[float], None] | None = None,
        on_audio_chunk: Callable[[np.ndarray], None] | None = None,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self._on_volume = on_volume
        self._on_audio_chunk = on_audio_chunk

        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Start recording audio from the microphone."""
        with self._lock:
            if self._stream is not None:
                raise RuntimeError("Recording is already in progress")

            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a Float32 array.

        Returns
        -------
        numpy.ndarray
            1-D float32 array of audio samples.
        """
        with self._lock:
            if self._stream is None:
                raise RuntimeError("Recording has not been started")

            self._stream.stop()
            self._stream.close()
            self._stream = None

            if not self._chunks:
                return np.empty(0, dtype=np.float32)

            audio = np.concatenate(self._chunks)
            self._chunks = []
            return audio.flatten()

    @property
    def is_recording(self) -> bool:
        """``True`` while the recorder is actively capturing audio."""
        with self._lock:
            return self._stream is not None

    # -- internals ------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice from the audio thread for every chunk."""
        if status:
            # Silently drop overflow/underflow info; could be logged later.
            pass
        # indata is only valid inside the callback, so copy.
        chunk = indata.copy()
        with self._lock:
            self._chunks.append(chunk)

        # Call audio chunk callback for realtime streaming
        if self._on_audio_chunk is not None:
            try:
                # Pass the chunk (flattened to 1D as expected by STT engines)
                self._on_audio_chunk(chunk.flatten())
            except Exception:
                pass  # Ignore errors in chunk callback

        # Calculate RMS volume and call the callback
        if self._on_volume is not None:
            try:
                rms = float(np.sqrt(np.mean(indata**2)))
                # Normalize to 0.0-1.0 range (typical speech RMS is 0.01-0.2)
                # Use a logarithmic-ish scaling for better visual feedback
                level = min(1.0, rms * 10.0)
                self._on_volume(level)
            except Exception:
                pass  # Ignore errors in volume callback


# -- module-level helpers -----------------------------------------------------


def normalize_audio(audio: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """Apply a gain boost and clip to [-1.0, 1.0].

    Whispered speech is typically very quiet.  A simple gain stage before
    sending audio to an STT model can significantly improve recognition.

    Parameters
    ----------
    audio:
        Float32 audio samples (any shape).
    gain:
        Multiplicative gain factor.

    Returns
    -------
    numpy.ndarray
        Amplified and clipped audio, same shape and dtype as *audio*.
    """
    return np.clip(audio * gain, -1.0, 1.0).astype(audio.dtype)
