from __future__ import annotations

import ctypes
import sys

from untype import clipboard


def test_inject_text_types_text_without_using_clipboard(monkeypatch) -> None:
    typed: list[str] = []
    copied: list[str] = []
    restored: list[str | None] = []

    monkeypatch.setattr(
        clipboard,
        "_type_text_into_foreground",
        lambda text: typed.append(text) or len(text),
    )
    monkeypatch.setattr(clipboard.pyperclip, "copy", lambda text: copied.append(text))
    monkeypatch.setattr(clipboard, "restore_clipboard", lambda text: restored.append(text))
    monkeypatch.setattr(clipboard.time, "sleep", lambda _seconds: None)

    result = clipboard.inject_text("hello", "original")

    assert result.delivery_method == "typed"
    assert result.typed_characters == 5
    assert result.copied_to_clipboard is False
    assert result.paste_simulated is False
    assert bool(result) is True
    assert typed == ["hello"]
    assert copied == []
    assert restored == []


def test_inject_text_reports_typing_failure(monkeypatch) -> None:
    monkeypatch.setattr(clipboard, "_type_text_into_foreground", lambda text: len(text) - 1)

    result = clipboard.inject_text("hello", "original")

    assert result.delivery_method == "typed"
    assert result.typed_characters == 4
    assert result.copied_to_clipboard is False
    assert result.paste_simulated is False
    assert bool(result) is False


def test_inject_text_treats_empty_string_as_success(monkeypatch) -> None:
    typed: list[str] = []
    monkeypatch.setattr(
        clipboard,
        "_type_text_into_foreground",
        lambda text: typed.append(text) or 0,
    )

    result = clipboard.inject_text("", "original")

    assert result.delivery_method == "typed"
    assert result.typed_characters == 0
    assert bool(result) is True
    assert typed == []


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


def test_type_text_into_foreground_types_each_character(monkeypatch) -> None:
    chars: list[str] = []

    monkeypatch.setattr(clipboard, "_type_character", lambda char: chars.append(char) or True)
    monkeypatch.setattr(clipboard.time, "sleep", lambda _seconds: None)

    typed = clipboard._type_text_into_foreground("ab")

    assert typed == 2
    assert chars == ["a", "b"]


def test_iter_utf16_code_units_supports_surrogate_pairs() -> None:
    assert clipboard._iter_utf16_code_units("🙂") == [0xD83D, 0xDE42]


def test_win32_input_structure_matches_sendinput_expectations() -> None:
    if sys.platform != "win32":
        return

    assert ctypes.sizeof(clipboard._INPUT) == 40
