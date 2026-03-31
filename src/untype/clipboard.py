"""Clipboard operations for the UnType interaction pipeline."""

from __future__ import annotations

import logging
import time
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


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of a clipboard-based text injection attempt."""

    copied_to_clipboard: bool
    paste_simulated: bool

    @property
    def delivered(self) -> bool:
        """Whether the app completed both clipboard copy and paste simulation."""
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
    """Inject *text* at the current cursor position via Ctrl+V.

    After pasting, the original clipboard content is restored so the user's
    clipboard is not clobbered.

    Returns:
        An :class:`InjectionResult` describing whether the text was copied to
        the clipboard and whether the paste hotkey was successfully simulated.
        The original clipboard is always restored regardless of the result.
    """
    copied_to_clipboard = False
    paste_simulated = False

    copied_to_clipboard = _copy_to_clipboard(text)

    if copied_to_clipboard:
        paste_simulated = _simulate_hotkey_with_retry(get_modifier_key(), "v")
        if paste_simulated:
            time.sleep(_PASTE_SETTLE_S)

    # Always restore the original clipboard, regardless of injection success
    restore_clipboard(original_clipboard)

    return InjectionResult(
        copied_to_clipboard=copied_to_clipboard,
        paste_simulated=paste_simulated,
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
