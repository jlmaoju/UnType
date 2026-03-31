from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from untype.stt import STTApiEngine


def test_audio_to_wav_produces_valid_mono_wav() -> None:
    engine = STTApiEngine("https://example.com/v1", "secret", sample_rate=16000)
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    wav_bytes = engine._audio_to_wav(audio)

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 3


def test_transcribe_posts_to_expected_endpoint(monkeypatch) -> None:
    engine = STTApiEngine(
        "https://example.com/v1",
        "secret",
        model="demo-model",
        language="en",
        sample_rate=16000,
    )
    audio = np.array([0.0, 0.25], dtype=np.float32)
    seen: dict[str, object] = {}

    class FakeResponse:
        text = '{"text": "hello world"}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"text": "hello world"}

    class FakeClient:
        def post(self, url: str, *, files, data):
            seen["url"] = url
            seen["files"] = files
            seen["data"] = data
            return FakeResponse()

    monkeypatch.setattr(engine, "_client", FakeClient())

    result = engine.transcribe(audio)

    assert result == "hello world"
    assert seen["url"] == "https://example.com/v1/audio/transcriptions"
    assert seen["data"] == {"model": "demo-model", "language": "en"}
    assert "file" in seen["files"]
    filename, wav_bytes, mime_type = seen["files"]["file"]
    assert filename == "audio.wav"
    assert mime_type == "audio/wav"
    assert isinstance(wav_bytes, bytes)


def test_transcribe_raises_on_malformed_response(monkeypatch) -> None:
    engine = STTApiEngine("https://example.com/v1", "secret")
    audio = np.array([0.0, 0.25], dtype=np.float32)

    class FakeResponse:
        text = '{"unexpected": "body"}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"unexpected": "body"}

    class FakeClient:
        def post(self, url: str, *, files, data):
            return FakeResponse()

    monkeypatch.setattr(engine, "_client", FakeClient())

    with pytest.raises(KeyError):
        engine.transcribe(audio)


def test_realtime_send_audio_noops_without_active_session() -> None:
    from untype.stt import STTRealtimeApiEngine

    engine = STTRealtimeApiEngine("secret")
    engine._session_ready.set()
    engine._recognition = None

    engine.send_audio(np.array([0.0, 0.25], dtype=np.float32))


def test_realtime_send_audio_for_active_session() -> None:
    from untype.stt import STTRealtimeApiEngine

    engine = STTRealtimeApiEngine("secret")
    sent_frames: list[bytes] = []

    class FakeRecognition:
        def send_audio_frame(self, frame: bytes) -> None:
            sent_frames.append(frame)

    engine._session_active = True
    engine._recognition = FakeRecognition()

    engine.send_audio(np.array([0.0, 0.25], dtype=np.float32))

    assert len(sent_frames) == 1
