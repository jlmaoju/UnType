from __future__ import annotations

from untype import clipboard


def test_inject_text_reports_success_and_restores_clipboard(monkeypatch) -> None:
    copied: list[str] = []
    simulated: list[tuple[object, str]] = []
    restored: list[str | None] = []

    monkeypatch.setattr(clipboard.pyperclip, "copy", lambda text: copied.append(text))
    monkeypatch.setattr(
        clipboard,
        "_simulate_hotkey",
        lambda key, char: simulated.append((key, char)),
    )
    monkeypatch.setattr(clipboard, "get_modifier_key", lambda: "ctrl")
    monkeypatch.setattr(clipboard, "restore_clipboard", lambda text: restored.append(text))
    monkeypatch.setattr(clipboard.time, "sleep", lambda _seconds: None)

    result = clipboard.inject_text("hello", "original")

    assert result.copied_to_clipboard is True
    assert result.paste_simulated is True
    assert bool(result) is True
    assert copied == ["hello"]
    assert simulated == [("ctrl", "v")]
    assert restored == ["original"]


def test_inject_text_reports_paste_failure_and_restores_clipboard(monkeypatch) -> None:
    restored: list[str | None] = []

    monkeypatch.setattr(clipboard.pyperclip, "copy", lambda text: None)
    monkeypatch.setattr(clipboard, "get_modifier_key", lambda: "ctrl")
    monkeypatch.setattr(
        clipboard,
        "_simulate_hotkey",
        lambda key, char: (_ for _ in ()).throw(RuntimeError("paste failed")),
    )
    monkeypatch.setattr(clipboard, "restore_clipboard", lambda text: restored.append(text))
    monkeypatch.setattr(clipboard.time, "sleep", lambda _seconds: None)

    result = clipboard.inject_text("hello", "original")

    assert result.copied_to_clipboard is True
    assert result.paste_simulated is False
    assert bool(result) is False
    assert restored == ["original"]


def test_inject_text_reports_copy_failure(monkeypatch) -> None:
    restored: list[str | None] = []

    def fail_copy(text: str) -> None:
        raise clipboard.pyperclip.PyperclipException("copy failed")

    monkeypatch.setattr(clipboard.pyperclip, "copy", fail_copy)
    monkeypatch.setattr(clipboard, "restore_clipboard", lambda text: restored.append(text))

    result = clipboard.inject_text("hello", "original")

    assert result.copied_to_clipboard is False
    assert result.paste_simulated is False
    assert bool(result) is False
    assert restored == ["original"]


def test_grab_selected_text_polls_until_clipboard_updates(monkeypatch) -> None:
    clipboard_values = iter(["", "", "picked text"])

    monkeypatch.setattr(clipboard, "save_clipboard", lambda: "original")
    monkeypatch.setattr(clipboard, "_copy_to_clipboard", lambda text: True)
    monkeypatch.setattr(clipboard, "_simulate_hotkey_with_retry", lambda key, char: True)
    monkeypatch.setattr(clipboard, "get_modifier_key", lambda: "ctrl")
    monkeypatch.setattr(clipboard.pyperclip, "paste", lambda: next(clipboard_values))
    monkeypatch.setattr(clipboard.time, "sleep", lambda _seconds: None)

    selected, original = clipboard.grab_selected_text()

    assert selected == "picked text"
    assert original == "original"


def test_simulate_hotkey_with_retry_retries_once(monkeypatch) -> None:
    attempts: list[tuple[object, str]] = []

    def flaky_hotkey(key: object, char: str) -> None:
        attempts.append((key, char))
        if len(attempts) == 1:
            raise RuntimeError("first attempt failed")

    monkeypatch.setattr(clipboard, "_simulate_hotkey", flaky_hotkey)
    monkeypatch.setattr(clipboard.time, "sleep", lambda _seconds: None)

    assert clipboard._simulate_hotkey_with_retry("ctrl", "v") is True
    assert attempts == [("ctrl", "v"), ("ctrl", "v")]
