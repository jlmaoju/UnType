"""Clipboard operations for the UnType interaction pipeline."""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

import pyperclip
from pynput.keyboard import Controller, Key

from untype.platform import get_modifier_key

logger = logging.getLogger(__name__)

_keyboard = Controller()

_CLIPBOARD_RETRY_COUNT = 2
_CLIPBOARD_RETRY_DELAY_S = 0.05
_SELECTION_POLL_COUNT = 5
_SELECTION_POLL_DELAY_S = 0.03
_SELECTION_COPY_SETTLE_S = 0.03
_PASTE_SETTLE_S = 0.1
_HOTKEY_ATTEMPTS = 2
_HOTKEY_RETRY_DELAY_S = 0.03
_MODIFIER_RELEASE_DELAY_S = 0.05
_HOTKEY_MODIFIER_PRESS_DELAY_S = 0.05
_HOTKEY_KEY_PRESS_DELAY_S = 0.02
_TYPE_CHARACTER_DELAY_S = 0.004

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004


if sys.platform == "win32":
    _ULONG_PTR = wintypes.WPARAM

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        ]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [
            ("mi", _MOUSEINPUT),
            ("ki", _KEYBDINPUT),
            ("hi", _HARDWAREINPUT),
        ]

    class _INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", _INPUTUNION),
        ]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
    _user32.SendInput.restype = wintypes.UINT


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of a text injection attempt."""

    copied_to_clipboard: bool
    paste_simulated: bool
    typed_characters: int = 0
    delivery_method: str = "clipboard"
    succeeded: bool | None = None

    @property
    def delivered(self) -> bool:
        """Whether the text was successfully delivered."""
        if self.succeeded is not None:
            return self.succeeded
        return self.copied_to_clipboard and self.paste_simulated

    def __bool__(self) -> bool:
        return self.delivered


def save_clipboard() -> str | None:
    """Save and return current clipboard text content."""
    try:
        return pyperclip.paste()
    except pyperclip.PyperclipException:
        return None


def restore_clipboard(content: str | None) -> None:
    """Restore clipboard to previous content after a short delay."""
    time.sleep(_CLIPBOARD_RETRY_DELAY_S)
    if not _copy_to_clipboard(content):
        pass


def grab_selected_text() -> tuple[str | None, str | None]:
    """Try to grab currently selected text via Ctrl+C.

    Returns:
        A tuple of (selected_text, original_clipboard).
        selected_text is None when nothing was selected.
    """
    original = save_clipboard()

    # Clear the clipboard so we can detect whether Ctrl+C wrote anything new.
    if not _copy_to_clipboard(""):
        return None, original

    # Simulate Ctrl+C to copy the current selection.
    if not _simulate_hotkey_with_retry(get_modifier_key(), "c"):
        restore_clipboard(original)
        return None, original
    time.sleep(_SELECTION_COPY_SETTLE_S)

    # Poll for the new clipboard contents instead of relying on a single fixed delay.
    for _ in range(_SELECTION_POLL_COUNT):
        try:
            text = pyperclip.paste()
        except pyperclip.PyperclipException:
            text = ""
        if text:
            return text, original
        time.sleep(_SELECTION_POLL_DELAY_S)
    return None, original


def inject_text(text: str, original_clipboard: str | None) -> InjectionResult:
    """Inject *text* at the current cursor position by typing it.

    This avoids clipboard-based paste semantics, which are unreliable in
    remote-control environments and can paste the remote machine's clipboard
    contents instead of the local text.
    """
    del original_clipboard

    if not text:
        return InjectionResult(
            copied_to_clipboard=False,
            paste_simulated=False,
            typed_characters=0,
            delivery_method="typed",
            succeeded=True,
        )

    typed_characters = _type_text_into_foreground(text)
    return InjectionResult(
        copied_to_clipboard=False,
        paste_simulated=False,
        typed_characters=typed_characters,
        delivery_method="typed",
        succeeded=typed_characters == len(text),
    )


def _copy_to_clipboard(content: str | None) -> bool:
    """Best-effort clipboard copy with a couple of short retries."""
    clipboard_text = "" if content is None else content
    for attempt in range(1, _CLIPBOARD_RETRY_COUNT + 1):
        try:
            pyperclip.copy(clipboard_text)
            return True
        except pyperclip.PyperclipException as exc:
            logger.warning(
                "Failed to copy text to clipboard (attempt %d/%d): %s",
                attempt,
                _CLIPBOARD_RETRY_COUNT,
                exc,
            )
            if attempt < _CLIPBOARD_RETRY_COUNT:
                time.sleep(_CLIPBOARD_RETRY_DELAY_S)
    return False


def _simulate_hotkey_with_retry(key: Key, char: str) -> bool:
    """Best-effort hotkey simulation with a very small bounded retry budget."""
    for attempt in range(1, _HOTKEY_ATTEMPTS + 1):
        try:
            _simulate_hotkey(key, char)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to simulate %s+%s (attempt %d/%d): %s",
                key,
                char,
                attempt,
                _HOTKEY_ATTEMPTS,
                exc,
            )
            if attempt < _HOTKEY_ATTEMPTS:
                time.sleep(_HOTKEY_RETRY_DELAY_S)
    return False


def _simulate_hotkey(key: Key, char: str) -> None:
    """Simulate a modifier+key hotkey combo (e.g. Ctrl+C, Ctrl+V).

    Any physically held modifiers (Alt, Shift, etc.) are released first so
    they don't contaminate the simulated combo.  The OS will naturally
    restore their state when the user physically releases them.
    """
    # Release any modifiers the user might still be holding from the hotkey
    _release_all_modifiers()
    time.sleep(_MODIFIER_RELEASE_DELAY_S)
    _keyboard.press(key)
    time.sleep(_HOTKEY_MODIFIER_PRESS_DELAY_S)
    _keyboard.press(char)
    time.sleep(_HOTKEY_KEY_PRESS_DELAY_S)
    _keyboard.release(char)
    time.sleep(_HOTKEY_KEY_PRESS_DELAY_S)
    _keyboard.release(key)


def _type_text_into_foreground(text: str) -> int:
    """Type *text* into the current foreground target one character at a time."""
    _release_all_modifiers()
    time.sleep(_MODIFIER_RELEASE_DELAY_S)

    typed = 0
    for char in text:
        if not _type_character(char):
            logger.warning("Failed to type character at index %d", typed)
            break
        typed += 1
        time.sleep(_TYPE_CHARACTER_DELAY_S)
    return typed


def _type_character(char: str) -> bool:
    """Type a single character, including Unicode and line breaks."""
    if char == "\r":
        return True
    if char == "\n":
        return _tap_special_key(Key.enter)
    if char == "\t":
        return _tap_special_key(Key.tab)

    if sys.platform == "win32":
        return _send_unicode_character(char)

    try:
        _keyboard.type(char)
    except Exception as exc:
        logger.warning("Failed to type character %r: %s", char, exc)
        return False
    return True


def _tap_special_key(key: Key) -> bool:
    """Press and release a non-text key such as Enter or Tab."""
    try:
        _keyboard.press(key)
        time.sleep(_HOTKEY_KEY_PRESS_DELAY_S)
        _keyboard.release(key)
    except Exception as exc:
        logger.warning("Failed to tap special key %s: %s", key, exc)
        return False
    return True


def _send_unicode_character(char: str) -> bool:
    """Send a Unicode character to the foreground app using Win32 SendInput."""
    for code_unit in _iter_utf16_code_units(char):
        if not _send_unicode_code_unit(code_unit, keyup=False):
            return False
        if not _send_unicode_code_unit(code_unit, keyup=True):
            return False
    return True


def _iter_utf16_code_units(text: str) -> list[int]:
    """Split *text* into UTF-16 code units for Win32 Unicode input."""
    encoded = text.encode("utf-16-le")
    return [int.from_bytes(encoded[i : i + 2], "little") for i in range(0, len(encoded), 2)]


def _send_unicode_code_unit(code_unit: int, *, keyup: bool) -> bool:
    """Send a single UTF-16 code unit with KEYEVENTF_UNICODE."""
    if sys.platform != "win32":
        return False

    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if keyup else 0)
    event = _INPUT(
        type=_INPUT_KEYBOARD,
        u=_INPUTUNION(
            ki=_KEYBDINPUT(
                wVk=0,
                wScan=code_unit,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )
    sent = _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT))
    if sent != 1:
        logger.warning(
            "SendInput failed for code unit %#x (keyup=%s, error=%d)",
            code_unit,
            keyup,
            ctypes.get_last_error(),
        )
        return False
    return True


def _release_all_modifiers() -> None:
    """Send key-up events for all common modifier keys."""
    for mod in (
        Key.alt_l,
        Key.alt_r,
        Key.ctrl_l,
        Key.ctrl_r,
        Key.shift_l,
        Key.shift_r,
        Key.cmd_l,
        Key.cmd_r,
    ):
        try:
            _keyboard.release(mod)
        except Exception:
            pass


# Public alias for use by other modules (e.g. ghost menu revert/regenerate).
release_all_modifiers = _release_all_modifiers
