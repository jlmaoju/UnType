"""Clipboard operations for the UnType interaction pipeline."""

from __future__ import annotations

import logging
import time

import pyperclip
from pynput.keyboard import Controller, Key

from untype.platform import get_modifier_key

logger = logging.getLogger(__name__)

_keyboard = Controller()


def save_clipboard() -> str | None:
    """Save and return current clipboard text content."""
    try:
        return pyperclip.paste()
    except pyperclip.PyperclipException:
        return None


def restore_clipboard(content: str | None) -> None:
    """Restore clipboard to previous content after a short delay."""
    time.sleep(0.05)
    try:
        if content is None:
            pyperclip.copy("")
        else:
            pyperclip.copy(content)
    except pyperclip.PyperclipException:
        pass


def grab_selected_text() -> tuple[str | None, str | None]:
    """Try to grab currently selected text via Ctrl+C.

    Returns:
        A tuple of (selected_text, original_clipboard).
        selected_text is None when nothing was selected.
    """
    original = save_clipboard()

    # Clear the clipboard so we can detect whether Ctrl+C wrote anything new.
    try:
        pyperclip.copy("")
    except pyperclip.PyperclipException:
        return None, original

    # Simulate Ctrl+C to copy the current selection.
    _simulate_hotkey(get_modifier_key(), "c")
    time.sleep(0.15)

    # Read whatever ended up on the clipboard.
    try:
        text = pyperclip.paste()
    except pyperclip.PyperclipException:
        return None, original

    if text:
        return text, original
    return None, original


def inject_text(text: str, original_clipboard: str | None) -> bool:
    """Inject *text* at the current cursor position via Ctrl+V.

    After pasting, the original clipboard content is restored so the user's
    clipboard is not clobbered.

    Returns:
        True if the text was successfully copied to clipboard (does not
        verify actual paste success). The original clipboard is always
        restored regardless of the result.
    """
    injection_succeeded = False

    try:
        pyperclip.copy(text)
        injection_succeeded = True
    except pyperclip.PyperclipException as e:
        logger.warning("Failed to copy text to clipboard: %s", e)

    if injection_succeeded:
        try:
            time.sleep(0.05)
            _simulate_hotkey(get_modifier_key(), "v")
            time.sleep(0.1)
        except Exception as e:
            logger.warning("Failed to simulate Ctrl+V: %s", e)

    # Always restore the original clipboard, regardless of injection success
    restore_clipboard(original_clipboard)

    return injection_succeeded


def _simulate_hotkey(key: Key, char: str) -> None:
    """Simulate a modifier+key hotkey combo (e.g. Ctrl+C, Ctrl+V).

    Any physically held modifiers (Alt, Shift, etc.) are released first so
    they don't contaminate the simulated combo.  The OS will naturally
    restore their state when the user physically releases them.
    """
    # Release any modifiers the user might still be holding from the hotkey
    _release_all_modifiers()
    time.sleep(0.05)
    _keyboard.press(key)
    time.sleep(0.05)
    _keyboard.press(char)
    time.sleep(0.02)
    _keyboard.release(char)
    time.sleep(0.02)
    _keyboard.release(key)


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
