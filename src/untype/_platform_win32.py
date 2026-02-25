"""Windows implementation of platform-specific operations (ctypes + user32)."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import threading
import tkinter as tk
from typing import Callable

from pynput.keyboard import Key

from untype.platform import CaretPosition, WindowIdentity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ctypes structures
# ---------------------------------------------------------------------------

user32 = ctypes.windll.user32  # type: ignore[attr-defined]
kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.wintypes.LONG), ("y", ctypes.wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.wintypes.LONG),
        ("top", ctypes.wintypes.LONG),
        ("right", ctypes.wintypes.LONG),
        ("bottom", ctypes.wintypes.LONG),
    ]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("hwndActive", ctypes.wintypes.HWND),
        ("hwndFocus", ctypes.wintypes.HWND),
        ("hwndCapture", ctypes.wintypes.HWND),
        ("hwndMenuOwner", ctypes.wintypes.HWND),
        ("hwndMoveSize", ctypes.wintypes.HWND),
        ("hwndCaret", ctypes.wintypes.HWND),
        ("rcCaret", RECT),
    ]


# ---------------------------------------------------------------------------
# Caret position
# ---------------------------------------------------------------------------


def get_caret_screen_position() -> CaretPosition:
    """Return the screen position of the text caret.

    Tries ``GetGUIThreadInfo`` + ``ClientToScreen`` first.  If the target
    application doesn't expose a Win32 caret (common with modern apps), falls
    back to the current mouse cursor position.
    """
    # Try GetGUIThreadInfo for the foreground thread.
    hwnd = user32.GetForegroundWindow()

    # Validate HWND before using it
    if not hwnd or not user32.IsWindow(hwnd):
        logger.debug("get_caret_screen_position: invalid HWND, falling back to mouse")
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return CaretPosition(x=pt.x, y=pt.y, found=False)

    tid = user32.GetWindowThreadProcessId(hwnd, None)
    if not tid:
        logger.debug("get_caret_screen_position: failed to get thread ID, falling back to mouse")
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return CaretPosition(x=pt.x, y=pt.y, found=False)

    gui = GUITHREADINFO()
    gui.cbSize = ctypes.sizeof(GUITHREADINFO)

    if user32.GetGUIThreadInfo(tid, ctypes.byref(gui)) and gui.hwndCaret:
        pt = POINT(gui.rcCaret.left, gui.rcCaret.top)
        if user32.ClientToScreen(gui.hwndCaret, ctypes.byref(pt)):
            return CaretPosition(x=pt.x, y=pt.y, found=True)

    # Fallback: mouse cursor position.
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return CaretPosition(x=pt.x, y=pt.y, found=False)


# ---------------------------------------------------------------------------
# Window tracking (Phase 2)
# ---------------------------------------------------------------------------


def get_foreground_window() -> WindowIdentity:
    """Take a snapshot of the current foreground window (HWND + PID + title).

    Returns an empty WindowIdentity (hwnd=0, title="", pid=0) if no valid
    foreground window exists.
    """
    hwnd = user32.GetForegroundWindow()

    # Validate HWND before using it
    if not hwnd or not user32.IsWindow(hwnd):
        logger.debug("get_foreground_window: invalid HWND returned")
        return WindowIdentity(hwnd=0, title="", pid=0)

    # Window title.
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value

    # Process ID.
    pid = ctypes.wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

    return WindowIdentity(hwnd=hwnd, title=title, pid=pid.value)


def verify_foreground_window(identity: WindowIdentity) -> bool:
    """Check whether the current foreground window matches *identity*.

    Compares both the HWND and the PID to handle the (rare) case where a
    window handle is reused by the OS for a different process.
    """
    current = get_foreground_window()
    return current.hwnd == identity.hwnd and current.pid == identity.pid


# ---------------------------------------------------------------------------
# Window styles (no-focus overlay)
# ---------------------------------------------------------------------------

GWL_EXSTYLE = -20
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008

GetWindowLongW = user32.GetWindowLongW
SetWindowLongW = user32.SetWindowLongW


def set_window_noactivate(tk_root: tk.Tk | tk.Toplevel) -> None:
    """Apply ``WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW`` to a tkinter window.

    This prevents the overlay from stealing focus when it appears and hides
    it from the taskbar / Alt+Tab list.
    """
    # Ensure the window has been realized so it has an HWND.
    tk_root.update_idletasks()
    hwnd = int(tk_root.wm_frame(), 16)

    style = GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
    SetWindowLongW(hwnd, GWL_EXSTYLE, style)


# ---------------------------------------------------------------------------
# Platform key
# ---------------------------------------------------------------------------


def get_modifier_key() -> Key:
    """Return the primary modifier key for keyboard shortcuts (Ctrl on Windows)."""
    return Key.ctrl_l


# ---------------------------------------------------------------------------
# Digit key interceptor (low-level keyboard hook)
# ---------------------------------------------------------------------------

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_QUIT = 0x0012
HC_ACTION = 0


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.wintypes.DWORD),
        ("scanCode", ctypes.wintypes.DWORD),
        ("flags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_long,  # return LRESULT
    ctypes.c_int,  # nCode
    ctypes.wintypes.WPARAM,  # wParam
    ctypes.wintypes.LPARAM,  # lParam
)

# Separate user32 instance for the digit interceptor so that argtypes
# set by pynput (which shares ctypes.windll.user32) don't conflict with
# our HOOKPROC definition.
_hook_user32 = ctypes.WinDLL("user32", use_last_error=True)
_hook_user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int,
    HOOKPROC,
    ctypes.wintypes.HINSTANCE,
    ctypes.wintypes.DWORD,
]
_hook_user32.SetWindowsHookExW.restype = ctypes.c_void_p
_hook_user32.CallNextHookEx.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.wintypes.WPARAM,
    ctypes.wintypes.LPARAM,
]
_hook_user32.CallNextHookEx.restype = ctypes.c_long
_hook_user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
_hook_user32.UnhookWindowsHookEx.restype = ctypes.wintypes.BOOL


class DigitKeyInterceptor:
    """Intercept digit keys 1-9 via a ``WH_KEYBOARD_LL`` hook.

    When *active*, key-down events for ``1``–``9`` are suppressed (never
    reach the target application) and ``on_digit(digit)`` is called.
    When *inactive*, all keys pass through normally.

    The hook runs on its own daemon thread with a Win32 message pump so
    that it satisfies the Windows requirement of pumping messages on the
    thread that installed the hook.
    """

    def __init__(self, on_digit: Callable[[int], None]) -> None:
        self._on_digit = on_digit
        self._active = False
        self._hook: int | None = None
        self._thread_id: int | None = None
        self._thread: threading.Thread | None = None
        # Must hold a reference to prevent garbage collection of the callback.
        self._hook_proc = HOOKPROC(self._low_level_handler)

    def start(self) -> None:
        """Start the hook thread (idempotent)."""
        if self._thread is not None:
            return
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(ready,),
            name="untype-digit-hook",
            daemon=True,
        )
        self._thread.start()
        ready.wait(timeout=5.0)

    def stop(self) -> None:
        """Post ``WM_QUIT`` to the hook thread's message pump and wait for cleanup."""
        tid = self._thread_id
        thread = self._thread
        if tid is not None:
            user32.PostThreadMessageW(tid, WM_QUIT, 0, 0)
            self._thread_id = None
            self._thread = None
            # Wait for the thread to actually terminate.
            if thread is not None:
                thread.join(timeout=1.0)

    def set_active(self, active: bool) -> None:
        self._active = active

    # -- internal ---------------------------------------------------------

    def _low_level_handler(self, nCode: int, wParam: int, lParam: int) -> int:
        if nCode == HC_ACTION and self._active:
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            if wParam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                if 0x31 <= kb.vkCode <= 0x39:  # VK_1 .. VK_9
                    digit = kb.vkCode - 0x30
                    try:
                        self._on_digit(digit)
                    except Exception:
                        logger.debug(
                            "on_digit callback error for digit %d",
                            digit,
                            exc_info=True,
                        )
                    return 1  # suppress this key event
        # Safe hook call with null check
        hook = self._hook
        if hook:
            return _hook_user32.CallNextHookEx(hook, nCode, wParam, lParam)
        return 0

    def _run(self, ready: threading.Event) -> None:
        """Thread entry: install hook, pump messages, unhook on WM_QUIT."""
        self._thread_id = kernel32.GetCurrentThreadId()
        self._hook = _hook_user32.SetWindowsHookExW(
            WH_KEYBOARD_LL,
            self._hook_proc,
            None,
            0,
        )
        if not self._hook:
            logger.error("SetWindowsHookExW failed for digit interceptor")
            ready.set()
            return

        logger.debug("Digit key interceptor hook installed (thread %d)", self._thread_id)
        ready.set()

        msg = ctypes.wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        _hook_user32.UnhookWindowsHookEx(self._hook)
        self._hook = None
        logger.debug("Digit key interceptor hook removed")
