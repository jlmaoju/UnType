"""macOS implementation of platform-specific operations (stub).

All functions raise ``NotImplementedError`` — macOS support is planned but
not yet implemented.
"""

from __future__ import annotations

import tkinter as tk

from pynput.keyboard import Key

from untype.platform import CaretPosition, WindowIdentity


def get_caret_screen_position() -> CaretPosition:
    raise NotImplementedError("macOS support coming soon")


def get_foreground_window() -> WindowIdentity:
    raise NotImplementedError("macOS support coming soon")


def verify_foreground_window(identity: WindowIdentity) -> bool:
    raise NotImplementedError("macOS support coming soon")


def set_window_noactivate(tk_root: tk.Tk | tk.Toplevel) -> None:
    raise NotImplementedError("macOS support coming soon")


def get_modifier_key() -> Key:
    raise NotImplementedError("macOS support coming soon")
