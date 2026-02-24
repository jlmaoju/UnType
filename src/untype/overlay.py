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

from untype.platform import set_window_noactivate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capsule dimensions (pill shape with text label)
# ---------------------------------------------------------------------------

_CAPSULE_W = 130
_CAPSULE_H = 36
_CAPSULE_R = 18  # half height → pill ends

# Capsule transparency range for alpha-breathing animation
_CAPSULE_ALPHA_MIN = 0.55
_CAPSULE_ALPHA_MAX = 0.85

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
        on_hold_inject: Callable[[], None] | None = None,
        on_hold_copy: Callable[[], None] | None = None,
    ) -> None:
        self._on_hold_inject = on_hold_inject
        self._on_hold_copy = on_hold_copy

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

        # Main capsule body (pill shape, monochrome).
        self._capsule_id = _draw_rounded_rect(
            canvas,
            0, 0, _CAPSULE_W, _CAPSULE_H, _CAPSULE_R,
            fill="#2a2a2a", outline="#555555", width=1,
        )

        # Status text (centred, small white label).
        self._text_id = canvas.create_text(
            _CAPSULE_W // 2, _CAPSULE_H // 2,
            text="", fill="#e0e0e0",
            font=("Segoe UI", 10),
            anchor="center",
        )

        self._canvas = canvas
        self._capsule_window = win

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
        elif op == "QUIT":
            self._do_quit()

    # ------------------------------------------------------------------
    # Command handlers (overlay thread only)
    # ------------------------------------------------------------------

    def _do_show(self, x: int, y: int, status: str) -> None:
        win = self._capsule_window
        if win is None:
            return

        self._capsule_at_corner = False
        self._pending_staging = None

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

        # Display label: strip trailing dots for a cleaner look.
        label = status.rstrip(".")
        canvas.itemconfigure(self._text_id, text=label)

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

    def _do_hide(self) -> None:
        self._animating = False
        self._flying = False
        self._capsule_at_corner = False
        win = self._capsule_window
        if win is not None:
            win.withdraw()
            win.attributes("-alpha", _CAPSULE_ALPHA_MAX)

    def _do_quit(self) -> None:
        self._animating = False
        self._flying = False
        root = self._root
        if root is not None:
            self._do_hide_hold_bubble()
            self._do_hide_staging()
            self._do_hide_recording_personas()
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
        """Fly the capsule to the bottom-right corner, keep it breathing."""
        if self._capsule_at_corner or self._flying:
            return  # already there or en route
        self._fly_bubble_text = None
        self._begin_fly()

    def _do_fly_to_hold_bubble(self, text: str) -> None:
        """Fly to corner then show bubble, or show bubble immediately
        if the capsule is already parked at the corner."""
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
        if self._rec_persona_window is not None:
            self._rec_persona_window.geometry(
                f"+{int(cx)}+{int(cy) + _CAPSULE_H + 4}"
            )

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
        canvas.create_text(
            _BUBBLE_W // 2, 4 + text_y + text_h + 4,
            text="L-click: inject | R-click: copy",
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
        """Create and show the recording persona bar below the capsule."""
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

        labels: list[tk.Label] = []
        for idx, (_pid, icon, name) in enumerate(personas):
            digit = idx + 1
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
            lbl.pack(side="left", padx=(0, 4))

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

        # Position below the capsule.  The capsule's top-left is placed by
        # _do_show at (x + 12, y - _CAPSULE_H - 8).
        capsule_x = x + 12
        capsule_y = y - _CAPSULE_H - 8

        # Keep on-screen (same logic as _do_show).
        screen_w = win.winfo_screenwidth()
        if capsule_x + _CAPSULE_W > screen_w:
            capsule_x = x - _CAPSULE_W - 12
        if capsule_y < 0:
            capsule_y = y + 24

        bar_x = capsule_x
        bar_y = capsule_y + _CAPSULE_H + 4

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
                shortcut_hint = "Ctrl+1 \u4eba\u683c"
            else:
                shortcut_hint = f"Ctrl+1~{n} \u4eba\u683c"
            hint_text = (
                "Enter \u6da6\u8272  \u2502  "
                f"{shortcut_hint}  \u2502  "
                "Shift+Enter \u76f4\u63a5\u53d1\u9001  \u2502  "
                "Esc \u53d6\u6d88"
            )
        else:
            hint_text = (
                "Enter \u6da6\u8272  \u2502  "
                "Shift+Enter \u76f4\u63a5\u53d1\u9001  \u2502  "
                "Esc \u53d6\u6d88"
            )
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
