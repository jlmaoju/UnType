"""Audio capture for whispered-speech input.

Provides push-to-talk style recording via sounddevice. All audio stays
in memory -- nothing ever touches disk.
"""

from __future__ import annotations

import threading
import time
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

    # Maximum recording duration in seconds (safety limit)
    MAX_RECORDING_SECONDS = 300  # 5 minutes

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
        self._start_time: float | None = None

    # -- public API -----------------------------------------------------------

    def start(self) -> None:
        """Start recording audio from the microphone.

        Raises
        ------
        RuntimeError
            If the configured device doesn't exist or is unavailable.
        """
        with self._lock:
            if self._stream is not None:
                raise RuntimeError("Recording is already in progress")

            # Validate device exists and is an input device
            self._validate_device()

            self._chunks = []
            self._start_time = time.time()
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
            except sd.PortAudioError as e:
                raise RuntimeError(
                    f"Failed to open audio device '{self.device or 'default'}': {e}"
                ) from e

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

            # np.concatenate of 1D arrays returns a 1D array, no need to flatten
            audio = np.concatenate(self._chunks)
            self._chunks = []
            return audio

    def abort(self) -> None:
        """Abort recording immediately without waiting for audio callback.

        This is used for cancellation scenarios where we want to stop
        recording as quickly as possible without blocking. Any audio
        already captured is discarded.
        """
        with self._lock:
            if self._stream is None:
                return

            try:
                # Close the stream directly - this aborts the audio callback immediately
                self._stream.close()
            except Exception:
                pass  # Ignore errors during abort
            self._stream = None
            self._chunks = []  # Discard any captured audio

    @property
    def is_recording(self) -> bool:
        """``True`` while the recorder is actively capturing audio."""
        with self._lock:
            return self._stream is not None

    def get_duration(self) -> float:
        """Get the current recording duration in seconds.

        Returns
        -------
        float
            Duration in seconds, or 0.0 if not recording.
        """
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time

    # -- internals ------------------------------------------------------------

    def _validate_device(self) -> None:
        """Validate that the configured device exists and is an input device.

        Raises
        ------
        RuntimeError
            If device doesn't exist or has no input channels.
        """
        if self.device is None:
            # None = system default, let sounddevice handle it
            return

        try:
            device_info = sd.query_devices(self.device)
        except (ValueError, sd.PortAudioError) as e:
            available = [
                sd.query_devices(i)["name"] for i in range(sd.query_devices(kind="input").size)
            ]
            available_list = ", ".join(available[:5])
            if len(available) > 5:
                available_list += "..."
            raise RuntimeError(
                f"Audio device '{self.device}' not found. Available devices: {available_list}"
            ) from e

        # Verify it's actually an input device
        if device_info["max_input_channels"] == 0:
            raise RuntimeError(
                f"Audio device '{device_info['name']}' has no input channels. "
                f"Please select a microphone device."
            )

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
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a numpy array")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    # Remove NaN/infinite values
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(audio.dtype)
