"""Floating capsule overlay — visual feedback during the voice-input pipeline.

Runs its own tkinter event loop on a dedicated daemon thread.  All public
methods are thread-safe (they enqueue commands that the tkinter thread drains
via ``after()`` polling).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import tkinter as tk
from typing import Callable

from untype.i18n import t
from untype.platform import set_window_noactivate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capsule dimensions (pill shape with text label)
# ---------------------------------------------------------------------------

_CAPSULE_W = 130
_CAPSULE_H = 36
_CAPSULE_R = 18  # half height → pill ends

# Capsule transparency range for alpha-breathing animation
# Set both to 0.9 for a constant 90% opacity (no breathing effect)
_CAPSULE_ALPHA_MIN = 0.9
_CAPSULE_ALPHA_MAX = 0.9

# Hold bubble dimensions
_BUBBLE_W = 300
_BUBBLE_H = 56
_BUBBLE_R = 16

# Staging area dimensions
_STAGING_W = 380
_STAGING_H = 160

# Transparent colour key (must not appear in the actual UI)
_TRANSPARENT_COLOR = "#010101"

# Animation timing
_POLL_INTERVAL_MS = 16  # ~60 fps queue polling
_ANIM_INTERVAL_MS = 33  # ~30 fps breathing animation
_BREATHING_PERIOD = 2.5  # seconds per full alpha cycle

# Fly-to-corner animation
_FLY_DURATION_MS = 650  # total flight time
_FLY_FRAME_MS = 12  # ~83 fps for ultra-smooth motion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out: slow start, fast middle, slow arrival."""
    if t < 0.5:
        return 4.0 * t * t * t
    p = -2.0 * t + 2.0
    return 1.0 - p * p * p / 2.0


# ---------------------------------------------------------------------------
# Capsule Overlay
# ---------------------------------------------------------------------------


class CapsuleOverlay:
    """Thread-safe floating capsule overlay.

    Parameters
    ----------
    on_hold_inject:
        Called (from overlay thread) when the user left-clicks the hold bubble.
    on_hold_copy:
        Called (from overlay thread) when the user right-clicks the hold bubble.
    """

    def __init__(
        self,
        capsule_position_mode: str = "fixed",
        capsule_fixed_x: int | None = None,
        capsule_fixed_y: int | None = None,
        on_position_changed: Callable[[int, int], None] | None = None,
        on_hold_inject: Callable[[], None] | None = None,
        on_hold_copy: Callable[[], None] | None = None,
        on_hold_ghost: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        on_ghost_revert: Callable[[], None] | None = None,
        on_ghost_regenerate: Callable[[], None] | None = None,
        on_ghost_use_raw: Callable[[], None] | None = None,
    ) -> None:
        self._capsule_position_mode = capsule_position_mode  # "caret" or "fixed"
        self._capsule_fixed_x = capsule_fixed_x
        self._capsule_fixed_y = capsule_fixed_y
        self._on_position_changed = on_position_changed
        self._on_hold_inject = on_hold_inject
        self._on_hold_copy = on_hold_copy
        self._on_hold_ghost = on_hold_ghost
        self._on_cancel = on_cancel
        self._on_ghost_revert = on_ghost_revert
        self._on_ghost_regenerate = on_ghost_regenerate
        self._on_ghost_use_raw = on_ghost_use_raw

        self._queue: queue.Queue[tuple] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._started = threading.Event()

        # Canvas items (set on overlay thread)
        self._canvas: tk.Canvas | None = None
        self._capsule_id: int | None = None
        self._text_id: int | None = None

        # Hold bubble state
        self._bubble_canvas: tk.Canvas | None = None
        self._bubble_window: tk.Toplevel | None = None

        # Staging area state (Phase 3)
        self._staging_window: tk.Toplevel | None = None
        self._staging_text_widget: tk.Text | None = None
        self._staging_event = threading.Event()
        self._staging_result_text: str = ""
        self._staging_result_action: str = ""  # "refine" | "raw" | "cancel"

        # Animation state
        self._anim_step: int = 0
        self._current_status: str = ""
        self._animating: bool = False

        # Fly-to-corner state
        self._flying: bool = False
        self._capsule_at_corner: bool = False
        self._fly_start_x: float = 0
        self._fly_start_y: float = 0
        self._fly_end_x: float = 0
        self._fly_end_y: float = 0
        self._fly_elapsed: float = 0
        self._fly_bubble_text: str | None = None
        # Deferred staging: if staging is requested while capsule is mid-flight,
        # the args are stashed here and executed when the flight lands.
        self._pending_staging: tuple | None = None

        # Recording persona bar state
        self._rec_persona_window: tk.Toplevel | None = None
        self._rec_persona_labels: list[tk.Label] = []
        self._rec_persona_on_click: Callable[[int], None] | None = None
        self._rec_persona_bar_w: int = 0  # measured width for right-alignment
        self._rec_persona_bar_h: int = 0  # measured height for above-capsule placement

        # Realtime preview state
        self._realtime_preview_window: tk.Toplevel | None = None
        self._realtime_preview_label: tk.Label | None = None
        self._realtime_preview_x: int = 0
        self._realtime_preview_y: int = 0

        # Cancel button state
        self._cancel_btn_id: int | None = None
        self._cancel_hover: bool = False
        self._volume_bar_id: int | None = None
        self._volume_bg_id: int | None = None

        # Ghost menu state
        self._ghost_window: tk.Toplevel | None = None
        self._ghost_expanded: bool = False
        self._ghost_hover_timer: str | None = None
        self._ghost_auto_dismiss_timer: str | None = None
        self._ghost_x: int = 0
        self._ghost_y: int = 0
        self._ghost_kb_listener: object | None = None
        self._ghost_mouse_listener: object | None = None

    # ------------------------------------------------------------------
    # Thread-safe public API (callable from any thread)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the overlay thread and wait until the tkinter root is ready."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="untype-overlay", daemon=True,
        )
        self._thread.start()
        self._started.wait(timeout=5.0)

    def stop(self) -> None:
        """Shut down the overlay thread."""
        self._queue.put(("QUIT",))

    def show(self, x: int, y: int, status: str) -> None:
        """Show the capsule at screen position (*x*, *y*) with *status*."""
        self._queue.put(("SHOW", x, y, status))

    def update_status(self, status: str) -> None:
        """Update the capsule status text and animation."""
        self._queue.put(("STATUS", status))

    def hide(self) -> None:
        """Hide the capsule."""
        self._queue.put(("HIDE",))

    def set_capsule_position_mode(self, mode: str) -> None:
        """Update the capsule position mode.

        Args:
            mode: One of "caret" (follow cursor) or "fixed" (draggable).
        """
        self._capsule_position_mode = mode

    def set_capsule_fixed_position(self, x: int, y: int) -> None:
        """Update the saved fixed position for fixed mode."""
        self._capsule_fixed_x = x
        self._capsule_fixed_y = y

    def update_volume(self, level: float) -> None:
        """Update the volume indicator in the capsule.

        Args:
            level: Audio level from 0.0 to 1.0.
        """
        self._queue.put(("VOLUME", level))

    def show_hold_bubble(self, text: str, x: int, y: int) -> None:
        """Show the hold-for-delivery bubble with a text preview."""
        self._queue.put(("HOLD_BUBBLE", text, x, y))

    def fly_to_corner(self) -> None:
        """Animate capsule flying to corner, keep it breathing there."""
        self._queue.put(("FLY_TO_CORNER",))

    def fly_to_hold_bubble(self, text: str) -> None:
        """Animate capsule flying to corner, then show hold bubble.

        If the capsule is already at the corner (from a prior ``fly_to_corner``),
        the bubble appears immediately without another flight.
        """
        self._queue.put(("FLY_TO_BUBBLE", text))

    def hide_hold_bubble(self) -> None:
        """Hide the hold bubble."""
        self._queue.put(("HIDE_BUBBLE",))

    def show_staging(
        self,
        text: str,
        x: int,
        y: int,
        at_corner: bool = False,
        personas: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Show the editable staging area with draft text.

        Hides the capsule first.  The staging area is a focusable window that
        accepts keyboard input (unlike the capsule).  If the capsule is
        mid-flight, the staging area is deferred until the flight lands.

        *personas* is an optional list of ``(id, icon, name)`` tuples for the
        persona bar.  When provided, clickable persona buttons and Ctrl+1/2/3
        shortcuts are added to the staging UI.
        """
        self._staging_event.clear()
        self._queue.put(("STAGING_SHOW", text, x, y, at_corner, personas))

    def wait_staging(self) -> tuple[str, str]:
        """Block until the user acts on the staging area.

        Returns ``(edited_text, action)`` where *action* is one of:

        - ``"refine"`` — Enter (default LLM refinement)
        - ``"persona:<id>"`` — persona-specific refinement
        - ``"raw"`` — Shift+Enter (inject raw)
        - ``"cancel"`` — Escape
        """
        self._staging_event.wait()
        return self._staging_result_text, self._staging_result_action

    # ------------------------------------------------------------------
    # Recording persona bar (thread-safe public API)
    # ------------------------------------------------------------------

    def show_recording_personas(
        self,
        personas: list[tuple[str, str, str]],
        x: int,
        y: int,
        on_click: Callable[[int], None] | None = None,
    ) -> None:
        """Show persona bar near the capsule during recording.

        Persists until :meth:`hide_recording_personas` is called.

        Parameters
        ----------
        personas:
            List of ``(id, icon, name)`` tuples.
        x, y:
            Screen position of the capsule (bar appears below it).
        on_click:
            Callback ``on_click(index)`` when a persona label is clicked.
        """
        self._queue.put(("REC_PERSONAS_SHOW", personas, x, y, on_click))

    def select_recording_persona(self, index: int) -> None:
        """Highlight the persona at *index* as pre-selected (-1 to clear)."""
        self._queue.put(("REC_PERSONAS_SELECT", index))

    def hide_recording_personas(self) -> None:
        """Hide the recording persona bar."""
        self._queue.put(("REC_PERSONAS_HIDE",))

    # ------------------------------------------------------------------
    # Realtime transcription preview (thread-safe public API)
    # ------------------------------------------------------------------

    def show_realtime_preview(self, x: int, y: int) -> None:
        """Show the realtime transcription preview window below the capsule.

        The preview displays streaming transcription text during recording.
        """
        self._queue.put(("REALTIME_PREVIEW_SHOW", x, y))

    def update_realtime_preview(self, text: str) -> None:
        """Update the text shown in the realtime preview window.

        Args:
            text: The current accumulated transcription text.
        """
        self._queue.put(("REALTIME_PREVIEW_UPDATE", text))

    def hide_realtime_preview(self) -> None:
        """Hide the realtime preview window."""
        self._queue.put(("REALTIME_PREVIEW_HIDE",))

    # ------------------------------------------------------------------
    # Ghost menu (thread-safe public API)
    # ------------------------------------------------------------------

    def show_ghost_menu(self, x: int, y: int) -> None:
        """Show the ghost menu icon near the caret after text injection.

        If capsule_position is fixed, the ghost menu is placed at a fixed
        position near the capsule instead of following the caret.
        """
        # Check if we should use a fixed position
        fixed_pos = self._get_ghost_fixed_position()
        if fixed_pos is not None:
            x, y = fixed_pos
        self._queue.put(("GHOST_SHOW", x, y))

    def _get_ghost_fixed_position(self) -> tuple[int, int] | None:
        """Calculate fixed ghost menu position based on capsule_position config.

        Returns None if ghost should follow caret (capsule_position_mode == "caret").
        """
        if self._capsule_position_mode != "fixed":
            return None

        # Get the capsule window's actual current position (not the saved config)
        # This ensures ghost follows the capsule when user drags it
        win = self._capsule_window
        if win is None:
            return None

        try:
            cx = win.winfo_x()
            cy = win.winfo_y()
        except Exception:
            return None

        # Position ghost to the right of the capsule
        gx = cx + _CAPSULE_W + 16
        gy = cy
        return (gx, gy)

        return None

    def hide_ghost_menu(self) -> None:
        """Hide and destroy the ghost menu."""
        self._queue.put(("GHOST_HIDE",))

    # ------------------------------------------------------------------
    # Overlay thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Entry point for the overlay thread — creates the tkinter root."""
        try:
            root = tk.Tk()
            root.withdraw()
            self._root = root

            self._setup_capsule_window(root)
            self._started.set()

            root.after(_POLL_INTERVAL_MS, self._poll_queue)
            root.mainloop()
        except Exception:
            logger.exception("Overlay thread crashed")
            self._started.set()  # unblock start() even on failure

    def _setup_capsule_window(self, root: tk.Tk) -> None:
        """Create the capsule Toplevel and canvas."""
        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-transparentcolor", _TRANSPARENT_COLOR)
        win.attributes("-alpha", _CAPSULE_ALPHA_MAX)
        win.geometry(f"{_CAPSULE_W}x{_CAPSULE_H}+0+0")
        win.withdraw()

        # Apply WS_EX_NOACTIVATE so this window never steals focus.
        try:
            set_window_noactivate(win)
        except Exception:
            logger.debug(
                "set_window_noactivate failed — overlay may steal focus",
            )

        canvas = tk.Canvas(
            win, width=_CAPSULE_W, height=_CAPSULE_H,
            bg=_TRANSPARENT_COLOR, highlightthickness=0, bd=0,
        )
        canvas.pack(fill="both", expand=True)

        # Main capsule body (pill shape, monochrome, stitch-edge dashed border).
        self._capsule_id = _draw_rounded_rect(
            canvas,
            2, 2, _CAPSULE_W - 2, _CAPSULE_H - 2, _CAPSULE_R - 2,
            fill="#2a2a2a", outline="#ffffff", width=2, dash=(14, 8),
        )

        # Volume bar background (bottom of capsule).
        vol_bar_y = _CAPSULE_H - 8
        vol_bar_h = 3
        self._volume_bg_id = canvas.create_rectangle(
            12, vol_bar_y, _CAPSULE_W - 12, vol_bar_y + vol_bar_h,
            fill="#1a1a1a", outline="",
        )
        # Volume bar foreground (filled based on level).
        self._volume_bar_id = canvas.create_rectangle(
            12, vol_bar_y, 12, vol_bar_y + vol_bar_h,
            fill="#4CAF50", outline="",
        )
        canvas.itemconfigure(self._volume_bar_id, state="hidden")
        canvas.itemconfigure(self._volume_bg_id, state="hidden")

        # Status text (centred, bold white label).
        self._text_id = canvas.create_text(
            _CAPSULE_W // 2, (_CAPSULE_H - 6) // 2,
            text="", fill="#e0e0e0",
            font=("Segoe UI", 11, "bold"),
            anchor="center",
        )

        # Cancel "×" button (right edge, initially hidden).
        # Position aligned with text (accounting for volume bar at bottom).
        self._cancel_btn_id = canvas.create_text(
            _CAPSULE_W - 14, (_CAPSULE_H - 6) // 2,
            text="\u00d7", fill="#666666",
            font=("Segoe UI", 13, "bold"),
            anchor="center",
            state="hidden",
        )
        canvas.tag_bind(self._cancel_btn_id, "<Button-1>", self._on_cancel_click)
        canvas.tag_bind(
            self._cancel_btn_id, "<Enter>",
            lambda e: (
                canvas.itemconfigure(self._cancel_btn_id, fill="#ff6666"),
                setattr(self, "_cancel_hover", True),
            ),
        )
        canvas.tag_bind(
            self._cancel_btn_id, "<Leave>",
            lambda e: (
                canvas.itemconfigure(self._cancel_btn_id, fill="#666666"),
                setattr(self, "_cancel_hover", False),
            ),
        )

        # Drag state for fixed mode
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0
        self._dragging: bool = False

        # Bind drag events to the capsule body (but not cancel button)
        canvas.tag_bind(self._capsule_id, "<Button-1>", self._on_drag_start)
        canvas.tag_bind(self._capsule_id, "<B1-Motion>", self._on_drag_motion)
        canvas.tag_bind(self._capsule_id, "<ButtonRelease-1>", self._on_drag_end)

        self._canvas = canvas
        self._capsule_window = win

    def _on_drag_start(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Start dragging the capsule (only in fixed mode)."""
        if self._capsule_position_mode != "fixed":
            return
        self._dragging = True
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root

    def _on_drag_motion(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Drag the capsule to new position."""
        if not self._dragging:
            return
        win = self._capsule_window
        if win is None:
            return

        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y

        # Get current position
        geom = win.geometry()
        # Parse geometry: "widthxheight+x+y"
        parts = geom.split("+")
        if len(parts) >= 3:
            current_x = int(parts[1])
            current_y = int(parts[2])
            new_x = current_x + dx
            new_y = current_y + dy

            # Keep on screen
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            new_x = max(0, min(new_x, screen_w - _CAPSULE_W))
            new_y = max(0, min(new_y, screen_h - _CAPSULE_H))

            win.geometry(f"+{new_x}+{new_y}")
            self._drag_start_x = event.x_root
            self._drag_start_y = event.y_root

    def _on_drag_end(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """End dragging and save the new position."""
        if not self._dragging:
            return
        self._dragging = False

        win = self._capsule_window
        if win is None:
            return

        # Get final position and save it
        geom = win.geometry()
        parts = geom.split("+")
        if len(parts) >= 3:
            new_x = int(parts[1])
            new_y = int(parts[2])
            self._capsule_fixed_x = new_x
            self._capsule_fixed_y = new_y

            # Notify main app to save to config
            if self._on_position_changed is not None:
                self._on_position_changed(new_x, new_y)

    def _on_cancel_click(self, event: object = None) -> None:
        """Handle click on the capsule cancel button."""
        if self._on_cancel is not None:
            threading.Thread(
                target=self._on_cancel,
                name="untype-cancel",
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        """Drain the command queue and dispatch each command."""
        root = self._root
        if root is None:
            return

        try:
            while True:
                cmd = self._queue.get_nowait()
                self._dispatch(cmd)
        except queue.Empty:
            pass

        root.after(_POLL_INTERVAL_MS, self._poll_queue)

    def _dispatch(self, cmd: tuple) -> None:
        op = cmd[0]
        if op == "SHOW":
            _, x, y, status = cmd
            self._do_show(x, y, status)
        elif op == "STATUS":
            _, status = cmd
            self._do_update_status(status)
        elif op == "HIDE":
            self._do_hide()
        elif op == "HOLD_BUBBLE":
            _, text, x, y = cmd
            self._do_show_hold_bubble(text, x, y)
        elif op == "FLY_TO_CORNER":
            self._do_fly_to_corner()
        elif op == "FLY_TO_BUBBLE":
            _, text = cmd
            self._do_fly_to_hold_bubble(text)
        elif op == "HIDE_BUBBLE":
            self._do_hide_hold_bubble()
        elif op == "STAGING_SHOW":
            _, text, x, y, at_corner, personas = cmd
            self._do_show_staging(text, x, y, at_corner, personas)
        elif op == "REC_PERSONAS_SHOW":
            _, personas, x, y, on_click = cmd
            self._do_show_recording_personas(personas, x, y, on_click)
        elif op == "REC_PERSONAS_SELECT":
            _, index = cmd
            self._do_select_recording_persona(index)
        elif op == "REC_PERSONAS_HIDE":
            self._do_hide_recording_personas()
        elif op == "REALTIME_PREVIEW_SHOW":
            _, x, y = cmd
            self._do_show_realtime_preview(x, y)
        elif op == "REALTIME_PREVIEW_UPDATE":
            _, text = cmd
            self._do_update_realtime_preview(text)
        elif op == "REALTIME_PREVIEW_HIDE":
            self._do_hide_realtime_preview()
        elif op == "GHOST_SHOW":
            _, x, y = cmd
            self._do_show_ghost_menu(x, y)
        elif op == "GHOST_HIDE":
            self._do_hide_ghost_menu()
        elif op == "VOLUME":
            _, level = cmd
            self._do_update_volume(level)
        elif op == "QUIT":
            self._do_quit()

    # ------------------------------------------------------------------
    # Command handlers (overlay thread only)
    # ------------------------------------------------------------------

    def _get_fixed_position(self) -> tuple[int, int] | None:
        """Calculate fixed capsule position based on config.

        Returns None if position should follow caret.
        """
        if self._capsule_position_mode != "fixed":
            return None

        root = self._root
        if root is None:
            return None

        # Use saved position if available, otherwise default to bottom center
        if self._capsule_fixed_x is not None and self._capsule_fixed_y is not None:
            return (self._capsule_fixed_x, self._capsule_fixed_y)

        # Default position: bottom center of screen
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        px = (screen_w - _CAPSULE_W) // 2
        py = screen_h - _CAPSULE_H - 80  # Above taskbar
        return (px, py)

    def _do_show(self, x: int, y: int, status: str) -> None:
        win = self._capsule_window
        if win is None:
            return

        self._capsule_at_corner = False
        self._pending_staging = None

        # Check for fixed position mode
        fixed_pos = self._get_fixed_position()
        if fixed_pos is not None:
            px, py = fixed_pos
        else:
            # Position the capsule: slightly above and to the right of the caret.
            px = x + 12
            py = y - _CAPSULE_H - 8

            # Keep on-screen.
            screen_w = win.winfo_screenwidth()
            if px + _CAPSULE_W > screen_w:
                px = x - _CAPSULE_W - 12
            if py < 0:
                py = y + 24

        win.attributes("-alpha", _CAPSULE_ALPHA_MAX)
        win.geometry(f"{_CAPSULE_W}x{_CAPSULE_H}+{px}+{py}")
        win.deiconify()
        win.lift()

        self._do_update_status(status)

    def _do_update_status(self, status: str) -> None:
        canvas = self._canvas
        if canvas is None:
            return

        self._current_status = status

        # Translate status for display (strip trailing dots for cleaner look).
        # Map internal status keys to i18n keys.
        status_key_map = {
            "Recording...": "overlay.recording",
            "Transcribing...": "overlay.transcribing",
            "Processing...": "overlay.processing",
            "Ready": "overlay.ready",
        }
        i18n_key = status_key_map.get(status, "")
        if i18n_key:
            label = t(i18n_key).rstrip(".")
        else:
            label = status.rstrip(".")
        canvas.itemconfigure(self._text_id, text=label)

        # Show/hide cancel button based on pipeline status.
        _CANCEL_STATUSES = (
            "Recording...", "Transcribing...", "Processing...",
        )
        show_cancel = status in _CANCEL_STATUSES
        # Text Y position: centered in the area above the volume bar
        text_y = (_CAPSULE_H - 6) // 2
        if self._cancel_btn_id is not None:
            if show_cancel:
                canvas.itemconfigure(self._cancel_btn_id, state="normal")
                # Shift main text slightly left to avoid overlap with X.
                canvas.coords(
                    self._text_id,
                    (_CAPSULE_W - 28) // 2, text_y,
                )
            else:
                canvas.itemconfigure(self._cancel_btn_id, state="hidden")
                canvas.coords(
                    self._text_id,
                    _CAPSULE_W // 2, text_y,
                )

        # Start/stop alpha-breathing animation.
        _ANIM_STATUSES = (
            "Recording...", "Transcribing...",
            "Processing...",
        )
        should_animate = status in _ANIM_STATUSES
        if should_animate and not self._animating:
            self._animating = True
            self._anim_step = 0
            self._animate_breathing()
        elif not should_animate:
            self._animating = False
            win = self._capsule_window
            if win is not None:
                win.attributes("-alpha", _CAPSULE_ALPHA_MAX)

        # Show/hide volume bar based on recording status.
        show_volume = status == "Recording..."
        if self._volume_bar_id is not None and self._volume_bg_id is not None:
            if show_volume:
                canvas.itemconfigure(self._volume_bg_id, state="normal")
                # Volume bar starts empty
                vol_bar_y = _CAPSULE_H - 8
                canvas.coords(self._volume_bar_id, 12, vol_bar_y, 12, vol_bar_y + 3)
            else:
                canvas.itemconfigure(self._volume_bar_id, state="hidden")
                canvas.itemconfigure(self._volume_bg_id, state="hidden")

    def _do_update_volume(self, level: float) -> None:
        """Update the volume bar based on audio level."""
        canvas = self._canvas
        if canvas is None:
            return

        # Only show during recording
        if self._current_status != "Recording...":
            return

        vol_bar_y = _CAPSULE_H - 8
        bar_width = _CAPSULE_W - 24  # 12px padding on each side
        filled_width = int(bar_width * min(1.0, max(0.0, level)))

        # Update the volume bar
        if self._volume_bar_id is not None:
            canvas.coords(
                self._volume_bar_id,
                12, vol_bar_y, 12 + filled_width, vol_bar_y + 3
            )
            canvas.itemconfigure(self._volume_bar_id, state="normal")

        # Color based on level (green -> yellow -> red)
        if self._volume_bar_id is not None:
            if level < 0.5:
                color = "#4CAF50"  # Green
            elif level < 0.8:
                color = "#FFC107"  # Yellow
            else:
                color = "#F44336"  # Red
            canvas.itemconfigure(self._volume_bar_id, fill=color)

    def _do_hide(self) -> None:
        self._animating = False
        self._flying = False
        self._capsule_at_corner = False
        # Hide cancel button.
        if self._cancel_btn_id is not None and self._canvas is not None:
            self._canvas.itemconfigure(self._cancel_btn_id, state="hidden")
        # Hide volume bar.
        if self._volume_bar_id is not None and self._canvas is not None:
            self._canvas.itemconfigure(self._volume_bar_id, state="hidden")
        if self._volume_bg_id is not None and self._canvas is not None:
            self._canvas.itemconfigure(self._volume_bg_id, state="hidden")
        win = self._capsule_window
        if win is not None:
            win.withdraw()
            win.attributes("-alpha", _CAPSULE_ALPHA_MAX)
        # Also hide realtime preview
        self._do_hide_realtime_preview()

    def _do_quit(self) -> None:
        self._animating = False
        self._flying = False
        root = self._root
        if root is not None:
            self._do_hide_hold_bubble()
            self._do_hide_staging()
            self._do_hide_recording_personas()
            self._do_hide_realtime_preview()
            self._do_hide_ghost_menu()
            # Unblock any pipeline thread waiting on staging.
            self._staging_result_action = "cancel"
            self._staging_event.set()
            root.quit()
            root.destroy()
            self._root = None

    # ------------------------------------------------------------------
    # Fly-to-corner animation
    # ------------------------------------------------------------------

    def _do_fly_to_corner(self) -> None:
        """Fly the capsule to the bottom-right corner, keep it breathing.

        If capsule_position_mode is fixed, skip the flight and just keep
        the capsule in place.
        """
        # Skip flight animation for fixed positions
        if self._capsule_position_mode != "caret":
            # Just mark as "at corner" so hold bubble logic works
            self._capsule_at_corner = True
            return

        if self._capsule_at_corner or self._flying:
            return  # already there or en route
        self._fly_bubble_text = None
        self._begin_fly()

    def _do_fly_to_hold_bubble(self, text: str) -> None:
        """Fly to corner then show bubble, or show bubble immediately
        if the capsule is already parked at the corner.

        For fixed positions, skip flight and show bubble directly.
        """
        # For fixed positions, just show the bubble in place
        if self._capsule_position_mode != "caret":
            win = self._capsule_window
            if win is not None:
                win.withdraw()
            self._capsule_at_corner = False
            self._do_show_hold_bubble(text, 0, 0)
            return

        if self._capsule_at_corner:
            # Already there — transform into bubble directly.
            self._animating = False
            self._capsule_at_corner = False
            win = self._capsule_window
            if win is not None:
                win.withdraw()
            self._do_show_hold_bubble(text, 0, 0)
            return
        if self._flying:
            # En route — upgrade: show bubble when it lands.
            self._fly_bubble_text = text
            return
        # Not flying, not at corner — start full flight.
        self._fly_bubble_text = text
        self._begin_fly()

    def _begin_fly(self) -> None:
        """Shared setup for all fly animations."""
        win = self._capsule_window
        root = self._root
        if win is None or root is None:
            return

        # Read current window position.
        win.update_idletasks()
        self._fly_start_x = float(win.winfo_x())
        self._fly_start_y = float(win.winfo_y())

        # Target: bottom-right corner of the screen.
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        self._fly_end_x = float(screen_w - _CAPSULE_W - 24)
        self._fly_end_y = float(screen_h - _CAPSULE_H - 80)

        self._fly_elapsed = 0
        self._flying = True

        root.after(_FLY_FRAME_MS, self._animate_fly)

    def _animate_fly(self) -> None:
        """Per-frame position update for the capsule flight."""
        if not self._flying:
            return

        win = self._capsule_window
        root = self._root
        if win is None or root is None:
            self._flying = False
            return

        self._fly_elapsed += _FLY_FRAME_MS
        raw_t = min(self._fly_elapsed / _FLY_DURATION_MS, 1.0)
        t = _ease_in_out_cubic(raw_t)

        # Interpolate position.
        cx = self._fly_start_x + (self._fly_end_x - self._fly_start_x) * t
        cy = self._fly_start_y + (self._fly_end_y - self._fly_start_y) * t
        win.geometry(f"+{int(cx)}+{int(cy)}")

        # Move recording persona bar in sync with the capsule.
        # Place ABOVE the capsule so it won't be covered by the taskbar.
        if self._rec_persona_window is not None:
            bar_x = int(cx) + _CAPSULE_W - self._rec_persona_bar_w
            bar_y = int(cy) - self._rec_persona_bar_h - 4
            self._rec_persona_window.geometry(f"+{bar_x}+{bar_y}")

        # Gentle alpha fade during the second half of the flight.
        if raw_t > 0.5:
            fade_t = (raw_t - 0.5) / 0.5
            alpha = _CAPSULE_ALPHA_MAX - fade_t * 0.25
            win.attributes("-alpha", alpha)
            # Fade persona bar in sync.
            if self._rec_persona_window is not None:
                self._rec_persona_window.attributes("-alpha", 0.9 - fade_t * 0.25)

        if raw_t >= 1.0:
            # Flight complete.
            self._flying = False

            # Deferred staging takes priority.
            if self._pending_staging is not None:
                args = self._pending_staging
                self._pending_staging = None
                self._animating = False
                win.withdraw()
                win.attributes("-alpha", _CAPSULE_ALPHA_MAX)
                self._do_show_staging(*args)
                return

            if self._fly_bubble_text is not None:
                # Result was ready — show bubble at destination.
                self._animating = False
                win.withdraw()
                win.attributes("-alpha", _CAPSULE_ALPHA_MAX)
                self._capsule_at_corner = False
                self._do_show_hold_bubble(
                    self._fly_bubble_text,
                    int(self._fly_end_x),
                    int(self._fly_end_y),
                )
                self._fly_bubble_text = None
            else:
                # No result yet — park here, keep breathing.
                self._capsule_at_corner = True
                win.attributes("-alpha", _CAPSULE_ALPHA_MAX)
                # Restore persona bar alpha after flight fade.
                if self._rec_persona_window is not None:
                    self._rec_persona_window.attributes("-alpha", 0.9)
                    self._rec_persona_window.lift()
            return

        root.after(_FLY_FRAME_MS, self._animate_fly)

    # ------------------------------------------------------------------
    # Hold bubble (Phase 2)
    # ------------------------------------------------------------------

    def _do_show_hold_bubble(self, text: str, x: int, y: int) -> None:
        # Hide the capsule first.
        self._do_hide()

        root = self._root
        if root is None:
            return

        # Destroy any existing bubble.
        self._do_hide_hold_bubble()

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-transparentcolor", _TRANSPARENT_COLOR)

        try:
            set_window_noactivate(win)
        except Exception:
            logger.debug("set_window_noactivate failed on hold bubble")

        # Max text width in pixels (bubble minus padding).
        text_max_px = _BUBBLE_W - 32

        # Truncate very long text.
        preview = text if len(text) <= 80 else text[:77] + "..."

        # Use a temporary canvas to measure text height.
        canvas = tk.Canvas(
            win, width=_BUBBLE_W, height=400,
            bg=_TRANSPARENT_COLOR, highlightthickness=0, bd=0,
        )
        canvas.pack(fill="both", expand=True)

        # Preview text.
        preview_id = canvas.create_text(
            _BUBBLE_W // 2, 0,
            text=preview, fill="white",
            font=("Segoe UI", 10),
            width=text_max_px, anchor="n",
        )
        bbox = canvas.bbox(preview_id)
        text_h = (bbox[3] - bbox[1]) if bbox else 16

        # Compute bubble height: padding + text + gap + hint + padding
        hint_h = 14
        bubble_h = max(_BUBBLE_H, 12 + text_h + 4 + hint_h + 12)

        # Reposition text items centred in the final bubble.
        text_y = (bubble_h - text_h - 4 - hint_h) // 2
        canvas.coords(preview_id, _BUBBLE_W // 2, 4 + text_y)

        # Hint text.
        hint_str = t("overlay.hold.hint")
        canvas.create_text(
            _BUBBLE_W // 2, 4 + text_y + text_h + 4,
            text=hint_str,
            fill="#888888",
            font=("Segoe UI", 8),
            width=text_max_px, anchor="n",
        )

        # Bubble body (drawn behind text — lower it).
        bg_id = _draw_rounded_rect(
            canvas,
            4, 4, _BUBBLE_W - 4, bubble_h - 4,
            _BUBBLE_R,
            fill="#2a2a2a", outline="#555555", width=1,
        )
        canvas.tag_lower(bg_id)

        # Resize canvas to fit.
        canvas.configure(height=bubble_h)

        # Click handlers.
        canvas.bind("<Button-1>", lambda e: self._on_bubble_left_click())
        canvas.bind("<Button-2>", lambda e: self._on_bubble_middle_click())
        canvas.bind("<Button-3>", lambda e: self._on_bubble_right_click())

        # Position near bottom-right of screen.
        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        px = screen_w - _BUBBLE_W - 24
        py = screen_h - bubble_h - 80

        win.geometry(f"{_BUBBLE_W}x{bubble_h}+{px}+{py}")
        win.deiconify()

        self._bubble_window = win
        self._bubble_canvas = canvas

    def _do_hide_hold_bubble(self) -> None:
        if self._bubble_window is not None:
            try:
                self._bubble_window.destroy()
            except Exception:
                pass
            self._bubble_window = None
            self._bubble_canvas = None

    def _on_bubble_left_click(self) -> None:
        self._do_hide_hold_bubble()
        if self._on_hold_inject:
            threading.Thread(
                target=self._on_hold_inject,
                name="untype-hold-inject",
                daemon=True,
            ).start()

    def _on_bubble_right_click(self) -> None:
        self._do_hide_hold_bubble()
        if self._on_hold_copy:
            threading.Thread(
                target=self._on_hold_copy,
                name="untype-hold-copy",
                daemon=True,
            ).start()

    def _on_bubble_middle_click(self) -> None:
        self._do_hide_hold_bubble()
        if self._on_hold_ghost:
            threading.Thread(
                target=self._on_hold_ghost,
                name="untype-hold-ghost",
                daemon=True,
            ).start()

    # ------------------------------------------------------------------
    # Recording persona bar (overlay thread only)
    # ------------------------------------------------------------------

    def _do_show_recording_personas(
        self,
        personas: list[tuple[str, str, str]],
        x: int,
        y: int,
        on_click: Callable[[int], None] | None,
    ) -> None:
        """Create and show the recording persona bar below the capsule.

        Layout adapts to persona count:
        - 1–2: single horizontal row
        - 3–4: 2×2 grid
        - 5–9: 3×3 grid
        """
        root = self._root
        if root is None:
            return

        # Destroy any existing persona bar.
        self._do_hide_recording_personas()

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.9)

        try:
            set_window_noactivate(win)
        except Exception:
            logger.debug("set_window_noactivate failed on recording persona bar")

        frame = tk.Frame(win, bg="#2a2a2a")
        frame.pack(fill="both", expand=True)

        # Determine grid columns based on persona count.
        n = len(personas)
        if n <= 2:
            cols = n  # 1 or 2 columns, single row
        elif n <= 4:
            cols = 2  # 2×2 grid
        else:
            cols = 3  # 3×3 grid

        labels: list[tk.Label] = []
        for idx, (_pid, icon, name) in enumerate(personas):
            digit = idx + 1
            row_i = idx // cols
            col_i = idx % cols
            lbl = tk.Label(
                frame,
                text=f" {digit} {icon} {name} ",
                font=("Segoe UI", 9),
                bg="#2a2a2a",
                fg="#888888",
                cursor="hand2",
                padx=2,
                pady=2,
            )
            lbl.grid(row=row_i, column=col_i, padx=(0, 4), pady=(0, 2), sticky="w")

            # Click binding.
            if on_click is not None:
                lbl.bind(
                    "<Button-1>",
                    lambda e, i=idx: self._on_rec_persona_label_click(i),
                )

            # Hover effects (only when not selected).
            lbl.bind(
                "<Enter>",
                lambda e, w=lbl: (
                    w.configure(fg="#cccccc")
                    if w.cget("bg") != "#4a4a8a"
                    else None
                ),
            )
            lbl.bind(
                "<Leave>",
                lambda e, w=lbl: (
                    w.configure(fg="#888888")
                    if w.cget("bg") != "#4a4a8a"
                    else None
                ),
            )

            labels.append(lbl)

        self._rec_persona_labels = labels
        self._rec_persona_on_click = on_click
        self._rec_persona_window = win

        # Position relative to the capsule's actual position.
        # Get capsule window's current position (works for both caret-following
        # and fixed positions).
        capsule_win = self._capsule_window
        if capsule_win is not None:
            capsule_win.update_idletasks()
            capsule_x = capsule_win.winfo_x()
            capsule_y = capsule_win.winfo_y()
        else:
            # Fallback: calculate from cursor position (legacy behavior)
            capsule_x = x + 12
            capsule_y = y - _CAPSULE_H - 8
            screen_w = win.winfo_screenwidth()
            if capsule_x + _CAPSULE_W > screen_w:
                capsule_x = x - _CAPSULE_W - 12
            if capsule_y < 0:
                capsule_y = y + 24

        bar_y = capsule_y + _CAPSULE_H + 4

        # Right-align the bar to the capsule's right edge so it doesn't
        # extend off-screen when the capsule is near the right margin.
        win.update_idletasks()
        bar_w = win.winfo_reqwidth()
        bar_h = win.winfo_reqheight()
        self._rec_persona_bar_w = bar_w
        self._rec_persona_bar_h = bar_h
        capsule_right = capsule_x + _CAPSULE_W
        bar_x = capsule_right - bar_w

        win.geometry(f"+{bar_x}+{bar_y}")
        win.deiconify()
        win.lift()

    def _on_rec_persona_label_click(self, index: int) -> None:
        """Handle click on a recording persona label."""
        cb = self._rec_persona_on_click
        if cb is not None:
            cb(index)

    def _do_select_recording_persona(self, index: int) -> None:
        """Highlight the persona at *index* (-1 to clear all)."""
        for i, lbl in enumerate(self._rec_persona_labels):
            if i == index:
                lbl.configure(bg="#4a4a8a", fg="#ffffff")
            else:
                lbl.configure(bg="#2a2a2a", fg="#888888")

    def _do_hide_recording_personas(self) -> None:
        """Destroy the recording persona bar window."""
        if self._rec_persona_window is not None:
            try:
                self._rec_persona_window.destroy()
            except Exception:
                pass
            self._rec_persona_window = None
            self._rec_persona_labels = []
            self._rec_persona_on_click = None
            self._rec_persona_bar_w = 0
            self._rec_persona_bar_h = 0
        # Also hide realtime preview when hiding personas
        self._do_hide_realtime_preview()

    # ------------------------------------------------------------------
    # Realtime transcription preview (appears below persona bar)
    # ------------------------------------------------------------------

    def _do_show_realtime_preview(self, x: int, y: int) -> None:
        """Create and show the realtime transcription preview window."""
        root = self._root
        if root is None:
            return

        # Destroy any existing preview window
        self._do_hide_realtime_preview()

        # Save capsule position for positioning the preview
        self._realtime_preview_x = x
        self._realtime_preview_y = y

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.85)

        try:
            set_window_noactivate(win)
        except Exception:
            logger.debug("set_window_noactivate failed on realtime preview")

        # Calculate position: below the persona bar (if visible) or capsule
        # Get actual capsule window position
        if self._capsule_window:
            try:
                cx = self._capsule_window.winfo_x()
                cy = self._capsule_window.winfo_y()
            except Exception:
                cx, cy = x, y
        else:
            cx, cy = x, y

        # Position below persona bar if visible, otherwise below capsule
        if self._rec_persona_window:
            # Position below the persona bar
            px = cx
            py = cy + _CAPSULE_H + 8 + self._rec_persona_bar_h + 4
        else:
            # Position directly below capsule
            px = cx
            py = cy + _CAPSULE_H + 8

        # Keep on-screen horizontally
        try:
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
        except Exception:
            # Fallback to reasonable defaults
            screen_w = 1920
            screen_h = 1080

        preview_w = 280
        if px + preview_w > screen_w:
            px = screen_w - preview_w - 8
        if px < 8:
            px = 8

        # Keep on-screen vertically
        if py + 80 > screen_h:
            py = screen_h - 80 - 8

        win.geometry(f"+{px}+{py}")

        # Configure window background
        win.configure(bg="#2a2a2a")

        # Create label for live text (directly in window, no canvas)
        label = tk.Label(
            win,
            text="",
            font=("Microsoft YaHei UI", 10),
            bg="#2a2a2a",
            fg="#cccccc",
            wraplength=preview_w - 16,
            justify="left",
            anchor="nw",
            padx=8,
            pady=8,
        )
        label.pack(fill="both", expand=True)

        self._realtime_preview_window = win
        self._realtime_preview_label = label

    def _do_update_realtime_preview(self, text: str) -> None:
        """Update the text in the realtime preview window."""
        if self._realtime_preview_label is not None:
            # Truncate if too long for display
            max_len = 100
            display_text = text if len(text) <= max_len else text[:max_len] + "..."
            self._realtime_preview_label.configure(text=display_text)

    def _do_hide_realtime_preview(self) -> None:
        """Destroy the realtime preview window."""
        if self._realtime_preview_window is not None:
            try:
                self._realtime_preview_window.destroy()
            except Exception:
                pass
            self._realtime_preview_window = None
            self._realtime_preview_label = None

    # ------------------------------------------------------------------
    # Staging area (Phase 3)
    # ------------------------------------------------------------------

    def _do_show_staging(
        self,
        text: str,
        x: int,
        y: int,
        at_corner: bool,
        personas: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Create and show the editable staging area."""
        # If capsule is mid-flight, defer until the flight lands so the
        # user sees the complete fly-to-corner animation.
        if self._flying:
            self._pending_staging = (text, x, y, at_corner, personas)
            return

        # Hide capsule and any existing bubble.
        self._do_hide()
        self._do_hide_hold_bubble()

        root = self._root
        if root is None:
            # Unblock waiter so the pipeline thread never hangs.
            self._staging_result_action = "cancel"
            self._staging_event.set()
            return

        # Destroy previous staging area if any.
        self._do_hide_staging()

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        # NOTE: intentionally no WS_EX_NOACTIVATE — staging must take focus.

        # Border frame (coloured outline around content).
        border = tk.Frame(win, bg="#555555", padx=1, pady=1)
        border.pack(fill="both", expand=True)

        inner = tk.Frame(border, bg="#2a2a2a")
        inner.pack(fill="both", expand=True)

        # Close "×" button (top-right corner).
        close_btn = tk.Label(
            inner,
            text="\u00d7",
            font=("Segoe UI", 12, "bold"),
            bg="#2a2a2a",
            fg="#666666",
            cursor="hand2",
            padx=4,
        )
        close_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
        close_btn.bind(
            "<Button-1>",
            lambda e: self._resolve_staging("cancel"),
        )
        close_btn.bind(
            "<Enter>",
            lambda e: close_btn.configure(fg="#ff6666"),
        )
        close_btn.bind(
            "<Leave>",
            lambda e: close_btn.configure(fg="#666666"),
        )

        # Editable text area.
        text_widget = tk.Text(
            inner,
            wrap="word",
            font=("Segoe UI", 11),
            bg="#1e1e1e",
            fg="#e0e0e0",
            insertbackground="#e0e0e0",
            selectbackground="#4a4a8a",
            selectforeground="#ffffff",
            relief="flat",
            padx=8,
            pady=8,
            height=4,
            width=1,  # minimal; pack fill="both" will expand
            undo=True,
        )
        text_widget.pack(
            fill="both", expand=True, padx=8, pady=(8, 4),
        )
        text_widget.insert("1.0", text)
        text_widget.mark_set("insert", "end-1c")

        # Persona bar (only when personas are provided).
        if personas:
            persona_frame = tk.Frame(inner, bg="#2a2a2a")
            persona_frame.pack(fill="x", padx=8, pady=(0, 4))

            for idx, (pid, icon, name) in enumerate(personas):
                label = tk.Label(
                    persona_frame,
                    text=f" {icon} {name} ",
                    font=("Segoe UI", 9),
                    bg="#3a3a3a",
                    fg="#c0c0c0",
                    cursor="hand2",
                    padx=4,
                    pady=2,
                )
                label.pack(side="left", padx=(0, 6))

                # Bind click to resolve with this persona.
                persona_action = f"persona:{pid}"
                label.bind(
                    "<Button-1>",
                    lambda e, a=persona_action: self._resolve_staging(a),
                )

                # Hover effect.
                label.bind(
                    "<Enter>",
                    lambda e, w=label: w.configure(bg="#4a4a4a", fg="#ffffff"),
                )
                label.bind(
                    "<Leave>",
                    lambda e, w=label: w.configure(bg="#3a3a3a", fg="#c0c0c0"),
                )

            # Bind Ctrl+1..9 on the text widget for keyboard persona selection.
            for idx, (pid, _, _) in enumerate(personas[:9]):
                key_num = str(idx + 1)
                persona_action = f"persona:{pid}"
                text_widget.bind(
                    f"<Control-Key-{key_num}>",
                    lambda e, a=persona_action: (
                        self._resolve_staging(a), "break"
                    )[-1],
                )

        # Hint bar.
        if personas:
            n = min(len(personas), 9)
            if n == 1:
                shortcut_hint = "Ctrl+1"
            else:
                shortcut_hint = f"Ctrl+1~{n}"
            hint_text = t("overlay.staging.hint_with_personas", shortcut=shortcut_hint)
        else:
            hint_text = t("overlay.staging.hint")
        hint = tk.Label(
            inner,
            text=hint_text,
            font=("Segoe UI", 9),
            bg="#2a2a2a",
            fg="#666666",
            anchor="center",
        )
        hint.pack(fill="x", padx=8, pady=(0, 8))

        # Key bindings (on text widget so we can return "break").
        text_widget.bind("<Return>", self._on_staging_enter)
        text_widget.bind("<Shift-Return>", self._on_staging_shift_enter)
        text_widget.bind("<Escape>", self._on_staging_escape)

        # Increase height when persona bar is present.
        staging_h = _STAGING_H + 32 if personas else _STAGING_H

        # Position.
        if at_corner:
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            px = screen_w - _STAGING_W - 24
            py = screen_h - staging_h - 80
        else:
            px = x + 12
            py = y + 24
            screen_w = win.winfo_screenwidth()
            screen_h = win.winfo_screenheight()
            if px + _STAGING_W > screen_w:
                px = x - _STAGING_W - 12
            if py + staging_h > screen_h:
                py = y - staging_h - 8

        win.geometry(f"{_STAGING_W}x{staging_h}+{px}+{py}")
        win.deiconify()
        win.lift()
        win.focus_force()
        text_widget.focus_force()
        # Retry focus after a brief delay (some WMs need time).
        win.after(100, lambda: text_widget.focus_force())

        self._staging_window = win
        self._staging_text_widget = text_widget

    def _on_staging_enter(self, event: object = None) -> str:
        """Enter pressed — refine via LLM."""
        self._resolve_staging("refine")
        return "break"

    def _on_staging_shift_enter(self, event: object = None) -> str:
        """Shift+Enter pressed — inject raw text, skip LLM."""
        self._resolve_staging("raw")
        return "break"

    def _on_staging_escape(self, event: object = None) -> str:
        """Escape pressed — cancel."""
        self._resolve_staging("cancel")
        return "break"

    def _resolve_staging(self, action: str) -> None:
        """Collect text from the widget, hide window, and signal the pipeline."""
        if self._staging_text_widget is not None and action != "cancel":
            self._staging_result_text = self._staging_text_widget.get(
                "1.0", "end-1c",
            ).strip()
        else:
            self._staging_result_text = ""
        self._staging_result_action = action
        self._do_hide_staging()
        self._staging_event.set()

    def _do_hide_staging(self) -> None:
        if self._staging_window is not None:
            try:
                self._staging_window.destroy()
            except Exception:
                pass
            self._staging_window = None
            self._staging_text_widget = None

    # ------------------------------------------------------------------
    # Ghost menu (post-injection undo/regenerate)
    # ------------------------------------------------------------------

    def _do_show_ghost_menu(self, x: int, y: int) -> None:
        """Create the ghost menu icon near the caret."""
        root = self._root
        if root is None:
            return

        # Destroy any existing ghost menu.
        self._do_hide_ghost_menu()

        self._ghost_x = x
        self._ghost_y = y

        win = tk.Toplevel(root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.75)

        try:
            set_window_noactivate(win)
        except Exception:
            logger.debug("set_window_noactivate failed on ghost menu")

        # Collapsed state: small icon button with thin white border.
        border_frame = tk.Frame(win, bg="#555555", padx=1, pady=1)
        border_frame.pack(fill="both", expand=True)

        frame = tk.Frame(border_frame, bg="#2a2a2a")
        frame.pack(fill="both", expand=True)

        icon_label = tk.Label(
            frame,
            text=t("ghost.icon"),
            font=("Segoe UI", 13),
            bg="#2a2a2a",
            fg="#888888",
            cursor="hand2",
            padx=4,
            pady=2,
        )
        icon_label.pack()

        # Menu frame (hidden initially, shown on hover).
        menu_frame = tk.Frame(frame, bg="#2a2a2a")
        # Not packed yet — expanded on hover.

        # Menu buttons.
        btn_use_raw = tk.Label(
            menu_frame,
            text=f" {t('ghost.raw')} ",
            font=("Segoe UI", 9),
            bg="#3a3a3a",
            fg="#c0c0c0",
            cursor="hand2",
            padx=4,
            pady=3,
        )
        btn_use_raw.pack(fill="x", padx=2, pady=(2, 1))
        btn_use_raw.bind(
            "<Button-1>",
            lambda e: self._on_ghost_action("use_raw"),
        )
        btn_use_raw.bind(
            "<Enter>", lambda e: btn_use_raw.configure(bg="#4a4a4a", fg="#ffffff"),
        )
        btn_use_raw.bind(
            "<Leave>", lambda e: btn_use_raw.configure(bg="#3a3a3a", fg="#c0c0c0"),
        )

        btn_regen = tk.Label(
            menu_frame,
            text=f" {t('ghost.redo')} ",
            font=("Segoe UI", 9),
            bg="#3a3a3a",
            fg="#c0c0c0",
            cursor="hand2",
            padx=4,
            pady=3,
        )
        btn_regen.pack(fill="x", padx=2, pady=(1, 1))
        btn_regen.bind(
            "<Button-1>",
            lambda e: self._on_ghost_action("regenerate"),
        )
        btn_regen.bind(
            "<Enter>", lambda e: btn_regen.configure(bg="#4a4a4a", fg="#ffffff"),
        )
        btn_regen.bind(
            "<Leave>", lambda e: btn_regen.configure(bg="#3a3a3a", fg="#c0c0c0"),
        )

        btn_revert = tk.Label(
            menu_frame,
            text=f" {t('ghost.edit')} ",
            font=("Segoe UI", 9),
            bg="#3a3a3a",
            fg="#c0c0c0",
            cursor="hand2",
            padx=4,
            pady=3,
        )
        btn_revert.pack(fill="x", padx=2, pady=(1, 1))
        btn_revert.bind(
            "<Button-1>",
            lambda e: self._on_ghost_action("revert"),
        )
        btn_revert.bind(
            "<Enter>", lambda e: btn_revert.configure(bg="#4a4a4a", fg="#ffffff"),
        )
        btn_revert.bind(
            "<Leave>", lambda e: btn_revert.configure(bg="#3a3a3a", fg="#c0c0c0"),
        )

        btn_dismiss = tk.Label(
            menu_frame,
            text=f" × {t('ghost.close')} ",
            font=("Segoe UI", 9),
            bg="#3a3a3a",
            fg="#c0c0c0",
            cursor="hand2",
            padx=4,
            pady=3,
        )
        btn_dismiss.pack(fill="x", padx=2, pady=(1, 2))
        btn_dismiss.bind(
            "<Button-1>",
            lambda e: self._do_hide_ghost_menu(),
        )
        btn_dismiss.bind(
            "<Enter>", lambda e: btn_dismiss.configure(bg="#4a4a4a", fg="#ff6666"),
        )
        btn_dismiss.bind(
            "<Leave>", lambda e: btn_dismiss.configure(bg="#3a3a3a", fg="#c0c0c0"),
        )

        self._ghost_expanded = False
        self._ghost_menu_frame = menu_frame

        # Hover to expand/collapse — bind on all widgets.
        all_hover_widgets = (
            win, border_frame, frame, icon_label,
            menu_frame, btn_use_raw, btn_regen, btn_revert, btn_dismiss,
        )
        for widget in all_hover_widgets:
            widget.bind("<Enter>", self._on_ghost_enter)
            widget.bind("<Leave>", self._on_ghost_leave)

        # Position: offset right and above the caret.
        px = x + 16
        py = y - 32

        # Keep on-screen.
        screen_w = win.winfo_screenwidth()
        if px + 28 > screen_w:
            px = x - 40
        if py < 0:
            py = y + 16

        win.geometry(f"+{px}+{py}")
        win.deiconify()
        win.lift()

        self._ghost_window = win

        # Auto-dismiss safety net: 30 seconds.
        self._ghost_auto_dismiss_timer = win.after(
            30_000, self._do_hide_ghost_menu,
        )

        # Start keyboard/mouse listeners to dismiss on any user activity.
        self._start_ghost_dismiss_listeners()

    def _on_ghost_enter(self, event: object = None) -> None:
        """Mouse entered the ghost area — expand menu."""
        # Cancel any pending collapse timer.
        if self._ghost_hover_timer is not None and self._ghost_window is not None:
            try:
                self._ghost_window.after_cancel(self._ghost_hover_timer)
            except Exception:
                pass
            self._ghost_hover_timer = None

        if not self._ghost_expanded and self._ghost_window is not None:
            self._ghost_expanded = True
            self._ghost_menu_frame.pack(fill="x")
            self._ghost_window.attributes("-alpha", 0.92)

    def _on_ghost_leave(self, event: object = None) -> None:
        """Mouse left the ghost area — schedule collapse after delay.

        Uses a generous delay to avoid flicker when moving between child
        widgets (each widget fires Leave→Enter on the sibling).
        """
        if self._ghost_window is not None:
            # Cancel any previous timer.
            if self._ghost_hover_timer is not None:
                try:
                    self._ghost_window.after_cancel(self._ghost_hover_timer)
                except Exception:
                    pass
            self._ghost_hover_timer = self._ghost_window.after(
                500, self._ghost_collapse,
            )

    def _ghost_collapse(self) -> None:
        """Collapse the ghost menu back to the icon."""
        self._ghost_hover_timer = None
        if self._ghost_expanded and self._ghost_window is not None:
            self._ghost_expanded = False
            self._ghost_menu_frame.pack_forget()
            self._ghost_window.attributes("-alpha", 0.75)

    def _on_ghost_action(self, action: str) -> None:
        """Handle a ghost menu button click."""
        self._do_hide_ghost_menu()
        if action == "use_raw" and self._on_ghost_use_raw is not None:
            threading.Thread(
                target=self._on_ghost_use_raw,
                name="untype-ghost-raw",
                daemon=True,
            ).start()
        elif action == "revert" and self._on_ghost_revert is not None:
            threading.Thread(
                target=self._on_ghost_revert,
                name="untype-ghost-revert",
                daemon=True,
            ).start()
        elif action == "regenerate" and self._on_ghost_regenerate is not None:
            threading.Thread(
                target=self._on_ghost_regenerate,
                name="untype-ghost-regen",
                daemon=True,
            ).start()

    def _do_hide_ghost_menu(self) -> None:
        """Destroy the ghost menu window and cancel timers/listeners."""
        self._stop_ghost_dismiss_listeners()
        if self._ghost_hover_timer is not None and self._ghost_window is not None:
            try:
                self._ghost_window.after_cancel(self._ghost_hover_timer)
            except Exception:
                pass
            self._ghost_hover_timer = None
        if self._ghost_auto_dismiss_timer is not None and self._ghost_window is not None:
            try:
                self._ghost_window.after_cancel(self._ghost_auto_dismiss_timer)
            except Exception:
                pass
            self._ghost_auto_dismiss_timer = None
        if self._ghost_window is not None:
            try:
                self._ghost_window.destroy()
            except Exception:
                pass
            self._ghost_window = None
            self._ghost_expanded = False

    # ------------------------------------------------------------------
    # Ghost dismiss listeners (keyboard/mouse activity detection)
    # ------------------------------------------------------------------

    def _start_ghost_dismiss_listeners(self) -> None:
        """Install temporary pynput listeners to dismiss ghost on user input."""
        from pynput import keyboard as kb
        from pynput import mouse as ms

        def on_key_press(key: object) -> bool:
            # Any keypress dismisses the ghost.
            self._queue.put(("GHOST_HIDE",))
            return False  # stop listener

        def on_mouse_click(
            x: int, y: int, button: object, pressed: bool,
        ) -> bool:
            if not pressed:
                return True  # ignore release
            # Check if the click is inside the ghost window area.
            # If so, let the ghost handle it (don't dismiss).
            win = self._ghost_window
            if win is not None:
                try:
                    wx = win.winfo_x()
                    wy = win.winfo_y()
                    ww = win.winfo_width()
                    wh = win.winfo_height()
                    # Generous padding to account for expanded menu.
                    if wx - 10 <= x <= wx + ww + 10 and wy - 10 <= y <= wy + wh + 80:
                        return True  # click on ghost — keep listening
                except Exception:
                    pass
            # Click outside ghost — dismiss.
            self._queue.put(("GHOST_HIDE",))
            return False  # stop listener

        kb_listener = kb.Listener(on_press=on_key_press)
        kb_listener.daemon = True
        kb_listener.start()
        self._ghost_kb_listener = kb_listener

        ms_listener = ms.Listener(on_click=on_mouse_click)
        ms_listener.daemon = True
        ms_listener.start()
        self._ghost_mouse_listener = ms_listener

    def _stop_ghost_dismiss_listeners(self) -> None:
        """Stop the temporary keyboard/mouse dismiss listeners."""
        if self._ghost_kb_listener is not None:
            try:
                self._ghost_kb_listener.stop()
            except Exception:
                pass
            self._ghost_kb_listener = None
        if self._ghost_mouse_listener is not None:
            try:
                self._ghost_mouse_listener.stop()
            except Exception:
                pass
            self._ghost_mouse_listener = None

    # ------------------------------------------------------------------
    # Breathing animation (alpha pulse)
    # ------------------------------------------------------------------

    def _animate_breathing(self) -> None:
        if not self._animating:
            return

        win = self._capsule_window
        root = self._root
        if win is None or root is None:
            return

        # Sinusoidal alpha pulse: fades between min and max transparency.
        t = (self._anim_step * _ANIM_INTERVAL_MS / 1000.0) / _BREATHING_PERIOD
        intensity = (math.sin(t * 2 * math.pi) + 1.0) / 2.0

        alpha = _CAPSULE_ALPHA_MIN + (
            _CAPSULE_ALPHA_MAX - _CAPSULE_ALPHA_MIN
        ) * intensity
        win.attributes("-alpha", alpha)

        self._anim_step += 1
        root.after(_ANIM_INTERVAL_MS, self._animate_breathing)


# ---------------------------------------------------------------------------
# Canvas helpers
# ---------------------------------------------------------------------------


def _draw_rounded_rect(
    canvas: tk.Canvas,
    x1: int, y1: int, x2: int, y2: int, r: int,
    **kwargs,
) -> int:
    """Draw a rounded rectangle on *canvas* and return its item id."""
    points = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
        x1 + r, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)
