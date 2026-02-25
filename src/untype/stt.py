"""Speech-to-text engine — local (faster-whisper), API (OpenAI-compatible),
and realtime API (Aliyun DashScope).
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import wave
from typing import Callable

import httpx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local STT engine (faster-whisper)
# ---------------------------------------------------------------------------


class STTEngine:
    """Low-latency local STT engine with preloaded Whisper model."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "zh",
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_threshold: float = 0.3,
    ):
        from faster_whisper import WhisperModel

        if device == "auto":
            device = self._detect_device()
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading Whisper model %s on %s (%s)...",
            model_size,
            device,
            compute_type,
        )

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._vad_threshold = vad_threshold

        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio buffer to text."""
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
            vad_parameters={"threshold": self._vad_threshold},
        )
        text = "".join(segment.text for segment in segments)
        logger.info("Transcription (%s, %.2fs): %s", info.language, info.duration, text)
        return text

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


# ---------------------------------------------------------------------------
# API STT engine (OpenAI-compatible /audio/transcriptions)
# ---------------------------------------------------------------------------


class STTApiEngine:
    """STT via OpenAI-compatible audio transcription API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-4o-transcribe",
        language: str = "zh",
        sample_rate: int = 16000,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language
        self._sample_rate = sample_rate
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
            verify=True,
        )
        logger.info("STT API engine ready (%s, model=%s)", self._base_url, self._model)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio buffer to text via API."""
        wav_bytes = self._audio_to_wav(audio)

        response = None
        try:
            response = self._client.post(
                f"{self._base_url}/audio/transcriptions",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": self._model, "language": self._language},
            )
            response.raise_for_status()
            data = response.json()
            text = data["text"]
            logger.info("API transcription: %s", text)
            return text
        except httpx.HTTPStatusError as exc:
            logger.error("STT API HTTP error %s: %s", exc.response.status_code, exc.response.text)
            raise
        except (KeyError, ValueError) as exc:
            body = response.text[:500] if response is not None and response.text else "(empty)"
            logger.error("Malformed STT API response (body: %s): %s", body, exc)
            raise
        except httpx.TimeoutException as exc:
            logger.error("STT API request timed out: %s", exc)
            raise

    def close(self) -> None:
        self._client.close()

    @property
    def is_loaded(self) -> bool:
        return True

    def _audio_to_wav(self, audio: np.ndarray) -> bytes:
        """Convert Float32 numpy audio to in-memory WAV bytes."""
        pcm16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Realtime API STT engine (Aliyun DashScope WebSocket)
# ---------------------------------------------------------------------------


class STTRealtimeApiEngine:
    """Streaming STT via Aliyun DashScope realtime recognition API.

    Uses WebSocket for low-latency streaming transcription.
    Audio is sent in chunks during recording, results arrive incrementally.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "paraformer-realtime-v2",
        language: str = "zh",
        format: str = "pcm",
        sample_rate: int = 16000,
        on_text_update: Callable[[str], None] | None = None,
    ):
        self._api_key = api_key
        self._model = model
        self._language = language
        self._format = format
        self._sample_rate = sample_rate
        self._on_text_update = on_text_update

        self._recognition = None
        self._result_queue: queue.Queue[str] = queue.Queue()
        self._finalized_text = ""  # Accumulated completed sentences
        self._pending_sentence = ""  # Current in-progress sentence
        self._current_text = ""
        self._lock = threading.Lock()
        self._session_active = False

        logger.info(
            "STT Realtime API engine ready (model=%s, format=%s, sr=%s)",
            self._model,
            self._format,
            self._sample_rate,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio buffer to text via realtime API.

        This sends the entire audio at once for compatibility with the existing
        pipeline interface. For true streaming, use start_session(), send_audio(),
        and stop_session() instead.
        """
        self.start_session()
        try:
            # Send audio in chunks (100ms = 1600 samples at 16kHz)
            chunk_size = int(0.1 * self._sample_rate)
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                self.send_audio(chunk)
            return self.get_result()
        finally:
            # Ensure session is always stopped, even if get_result() raises
            self.stop_session()

    def start_session(self) -> None:
        """Start a new realtime recognition session."""
        if self._session_active:
            logger.warning("Session already active, ignoring start_session")
            return

        try:
            import dashscope
            from dashscope.audio.asr import Recognition, RecognitionCallback

            dashscope.api_key = self._api_key

            # Build inner callback class inheriting from the SDK base class
            engine_ref = self

            class _Callback(RecognitionCallback):
                def on_open(self) -> None:
                    logger.debug("Realtime recognition WebSocket opened")

                def on_event(self, result) -> None:
                    try:
                        from dashscope.audio.asr import RecognitionResult

                        sentence = result.get_sentence()
                        if not isinstance(sentence, dict) or "text" not in sentence:
                            return
                        text = sentence["text"]
                        if not text:
                            return

                        is_end = RecognitionResult.is_sentence_end(sentence)

                        with engine_ref._lock:
                            if is_end:
                                # Sentence finalized — append to finalized, clear pending
                                engine_ref._finalized_text += text
                                engine_ref._pending_sentence = ""
                            else:
                                # In-progress update — replace pending sentence
                                engine_ref._pending_sentence = text
                            engine_ref._current_text = (
                                engine_ref._finalized_text + engine_ref._pending_sentence
                            )

                        display_text = engine_ref._current_text
                        if engine_ref._on_text_update:
                            engine_ref._on_text_update(display_text)
                        logger.debug("Realtime transcript update: %s", display_text)
                    except Exception as exc:
                        logger.warning("Error processing recognition event: %s", exc)

                def on_complete(self) -> None:
                    logger.debug("Realtime recognition completed")

                def on_error(self, message) -> None:
                    try:
                        request_id = getattr(message, "request_id", "unknown")
                        msg = getattr(message, "message", "Unknown error")
                        logger.error(
                            "Realtime recognition error (request_id=%s): %s",
                            request_id,
                            msg,
                        )
                    except Exception as exc:
                        logger.error("Realtime recognition error: %s", exc)

                def on_close(self) -> None:
                    logger.debug("Realtime recognition WebSocket closed")

            self._recognition = Recognition(
                model=self._model,
                format=self._format,
                sample_rate=self._sample_rate,
                callback=_Callback(),
            )
            self._recognition.start()
            self._session_active = True
            self._finalized_text = ""
            self._pending_sentence = ""
            self._current_text = ""
            logger.info("Realtime recognition session started")
        except ImportError as exc:
            logger.error("dashscope package not installed: %s", exc)
            raise RuntimeError(
                "dashscope package is required for realtime STT. "
                "Install with: pip install dashscope"
            ) from exc

    def send_audio(self, audio: np.ndarray) -> None:
        """Send an audio chunk to the ongoing recognition session.

        Args:
            audio: Float32 numpy array at the configured sample rate.
                   Recommended chunk size: 100ms (1600 samples at 16kHz).
        """
        if not self._session_active:
            logger.warning("No active session, cannot send audio")
            return

        # Convert to PCM16 bytes
        pcm16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        self._recognition.send_audio_frame(pcm16.tobytes())

    def start_recording_session(self) -> None:
        """Start a realtime recognition session for recording.

        This starts the WebSocket connection so audio can be streamed
        during recording via on_audio_chunk().
        """
        self.start_session()

    def on_audio_chunk(self, audio: np.ndarray) -> None:
        """Handle an audio chunk during recording (to be called from audio callback).

        This method accumulates chunks and sends them to the recognition service
        in batches (every ~100ms) for efficiency.

        Args:
            audio: Float32 numpy array audio chunk.
        """
        # Accumulate audio chunks and send periodically
        # For now, send each chunk directly (dashscope handles buffering)
        self.send_audio(audio)

    def stop_session(self) -> str:
        """Stop the recognition session and return the final result.

        Returns:
            The complete transcribed text.
        """
        if not self._session_active:
            return self._current_text

        self._recognition.stop()
        self._session_active = False
        logger.info("Realtime recognition session stopped: %s", self._current_text)
        return self._current_text

    def get_result(self) -> str:
        """Get the current accumulated transcription result."""
        with self._lock:
            return self._current_text

    def close(self) -> None:
        """Clean up resources."""
        if self._session_active:
            self.stop_session()
        self._recognition = None

    @property
    def is_loaded(self) -> bool:
        return True
