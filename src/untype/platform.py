"""Platform-specific operations — caret position, window tracking, window styles.

This module defines the public API and shared data classes, then dispatches to
the correct backend based on ``sys.platform``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared data classes (platform-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaretPosition:
    """Screen coordinates of the text caret (or mouse cursor as fallback)."""

    x: int
    y: int
    found: bool  # True if real caret was found, False if mouse fallback


@dataclass(frozen=True)
class WindowIdentity:
    """Snapshot of a foreground window for later verification."""

    hwnd: int
    title: str
    pid: int


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    from untype._platform_win32 import (
        get_caret_screen_position,
        get_foreground_window,
        get_modifier_key,
        set_window_noactivate,
        verify_foreground_window,
    )
elif sys.platform == "darwin":
    from untype._platform_darwin import (
        get_caret_screen_position,
        get_foreground_window,
        get_modifier_key,
        set_window_noactivate,
        verify_foreground_window,
    )
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")
