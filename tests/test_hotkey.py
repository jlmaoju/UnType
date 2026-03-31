from __future__ import annotations

import pytest
from pynput import keyboard

from untype.hotkey import parse_hotkey


def test_parse_hotkey_supports_modifiers_and_character_trigger() -> None:
    modifiers, trigger = parse_hotkey("ctrl+shift+a")

    assert modifiers == {"ctrl", "shift"}
    assert isinstance(trigger, keyboard.KeyCode)
    assert trigger.char == "a"


def test_parse_hotkey_normalizes_win_alias() -> None:
    modifiers, trigger = parse_hotkey("win+space")

    assert modifiers == {"cmd"}
    assert trigger == keyboard.Key.space


@pytest.mark.parametrize(
    "hotkey",
    [
        "",
        "ctrl",
        "meta+a",
        "ctrl+unknown",
    ],
)
def test_parse_hotkey_rejects_invalid_inputs(hotkey: str) -> None:
    with pytest.raises(ValueError):
        parse_hotkey(hotkey)
