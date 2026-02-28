"""System tray icon with settings dialog and status display."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable
from urllib.parse import urlparse

import pystray
from PIL import Image, ImageDraw

from untype.build_info import HAS_LOCAL_STT
from untype.config import AppConfig, save_config
from untype.i18n import get_locale_display_name, list_available_locales, t

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_url(url: str) -> bool:
    """Validate that a URL string is properly formatted.

    Empty strings are considered valid (optional field).
    """
    if not url:
        return True
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def _validate_gain(value: float) -> bool:
    """Validate that gain_boost is within acceptable range (0.1 to 10.0)."""
    return 0.1 <= value <= 10.0


# ---------------------------------------------------------------------------
# Icon colours for each application state
# ---------------------------------------------------------------------------

_STATUS_COLORS: dict[str, str] = {
    "Ready": "#4CAF50",  # green
    "Recording...": "#FF9800",  # orange
    "Transcribing...": "#2196F3",  # blue
    "Processing...": "#9C27B0",  # purple
    "Error": "#F44336",  # red
}

# Translation keys for status text
_STATUS_KEYS: dict[str, str] = {
    "Ready": "tray.status.ready",
    "Recording...": "tray.status.recording",
    "Transcribing...": "tray.status.transcribing",
    "Processing...": "tray.status.processing",
    "Error": "tray.status.error",
}

_DEFAULT_ICON_COLOR = "#4CAF50"


# ---------------------------------------------------------------------------
# Icon helper
# ---------------------------------------------------------------------------


def _create_icon_image(color: str = _DEFAULT_ICON_COLOR, size: int = 64) -> Image.Image:
    """Create a simple colored circle icon on a transparent background."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=color,
    )
    return image


# ---------------------------------------------------------------------------
# Settings dialog (tkinter)
# ---------------------------------------------------------------------------


class SettingsDialog:
    """Tkinter settings dialog for editing the UnType configuration.

    Because tkinter is *not* thread-safe, the dialog creates its own
    :class:`tkinter.Tk` root (hidden) and runs a local mainloop on the
    calling thread.  Callers should invoke :meth:`show` from a dedicated
    thread — **never** from the pystray callback thread directly.
    """

    def __init__(self, config: AppConfig, on_save: Callable[[AppConfig], None], on_rerun_wizard: Callable[[], None] | None = None) -> None:
        """
        Args:
            config: Current app configuration.
            on_save: Callback invoked when the user clicks *Save*;
                     receives the updated :class:`AppConfig`.
            on_rerun_wizard: Optional callback to rerun the setup wizard.
        """
        self._config = config
        self._on_save = on_save
        self._on_rerun_wizard = on_rerun_wizard

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        """Show the settings dialog (blocks until closed)."""
        root = tk.Tk()
        root.title(t("settings.title"))
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # Ensure the window appears in front and grabs focus
        root.after(100, lambda: root.attributes("-topmost", False))

        frame = ttk.Frame(root, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")

        row = 0

        # -- Hotkey -------------------------------------------------------
        row = self._heading(frame, t("settings.heading.hotkey"), row)
        hotkey_var = self._hotkey_field(
            root,
            frame,
            t("settings.trigger"),
            self._config.hotkey.trigger,
            row,
        )
        row += 1
        # Mode combo with translated options
        mode_options = ["toggle", "hold"]
        mode_labels = [t("settings.mode.toggle"), t("settings.mode.hold")]
        hotkey_mode_var = self._combo_field(
            root,
            frame,
            t("settings.mode"),
            self._config.hotkey.mode,
            mode_options,
            row,
            labels=mode_labels,
        )
        row += 1

        # -- LLM ----------------------------------------------------------
        row = self._heading(frame, t("settings.heading.llm"), row)
        llm_url_var = self._text_field(
            root,
            frame,
            t("settings.base_url"),
            self._config.llm.base_url,
            row,
        )
        row += 1
        llm_key_var = self._text_field(
            root,
            frame,
            t("settings.api_key"),
            self._config.llm.api_key,
            row,
            show="*",
        )
        row += 1
        llm_model_var = self._text_field(
            root,
            frame,
            t("settings.model"),
            self._config.llm.model,
            row,
        )
        row += 1

        # -- STT -----------------------------------------------------------
        row = self._heading(frame, t("settings.heading.stt"), row)

        # Build STT backend options based on available features
        if HAS_LOCAL_STT:
            stt_backend_options = ["api", "local", "realtime_api"]
            stt_backend_labels = [
                t("settings.backend.api"),
                t("settings.backend.local"),
                t("settings.backend.realtime_api"),
            ]
        else:
            # Online version: local STT is not available
            # If current backend is local, switch to api
            if self._config.stt.backend == "local":
                self._config.stt.backend = "api"

            stt_backend_options = ["api", "local", "realtime_api"]
            stt_backend_labels = [
                t("settings.backend.api"),
                t("settings.backend.local_disabled"),
                t("settings.backend.realtime_api"),
            ]

        # Store field label widgets for show/hide (labels and their input widgets)
        stt_field_labels: dict[str, tuple[ttk.Label, tk.Widget]] = {}
        # Store combobox widget reference for disabling
        stt_combobox_ref: dict[str, tk.Widget] = {}

        def _on_stt_backend_change(backend: str) -> None:
            """Show/hide STT fields based on backend selection."""
            # api fields: stt_api_url, stt_api_key, stt_api_model
            # local fields: stt_model (model_size), stt_device
            # realtime_api fields: stt_realtime_api_key, stt_realtime_api_model
            api_fields = [
                t("settings.stt_api_url"),
                t("settings.stt_api_key"),
                t("settings.stt_api_model"),
            ]
            local_fields = [t("settings.local_model"), t("settings.local_device")]
            realtime_fields = [
                t("settings.stt_realtime_api_key"),
                t("settings.stt_realtime_api_model"),
            ]

            # If local STT is not available and user tries to select it
            if backend == "local" and not HAS_LOCAL_STT:
                import tkinter.messagebox as messagebox
                messagebox.showinfo(
                    t("settings.backend.local") if hasattr(t, "_dict") else "Local Model",
                    t("tooltip.local_stt_disabled") if hasattr(t, "_dict") else "Local speech recognition requires the full version.\n\nPlease download UnType-v0.3.0-full.zip from GitHub releases."
                )
                # Reset to api
                stt_backend_var.set("api")
                # Trigger the change callback for api
                backend = "api"

            # Warn when selecting local mode (model download required)
            if backend == "local" and self._config.stt.backend != "local" and HAS_LOCAL_STT:
                import tkinter.messagebox as messagebox
                messagebox.showwarning(
                    "本地模型提示",
                    "本地模式需要下载 Whisper 模型（约 500MB），请确保网络环境正常。\n\n"
                    "首次使用时会自动从 HuggingFace 下载模型，可能需要几分钟时间。"
                )

            for label_text, (label_widget, input_widget) in stt_field_labels.items():
                should_show = (
                    (label_text in api_fields and backend == "api")
                    or (label_text in local_fields and backend == "local")
                    or (label_text in realtime_fields and backend == "realtime_api")
                )
                if should_show:
                    label_widget.grid()
                    input_widget.grid()
                else:
                    label_widget.grid_remove()
                    input_widget.grid_remove()

        stt_backend_var = self._combo_field(
            root,
            frame,
            t("settings.backend"),
            self._config.stt.backend,
            stt_backend_options,
            row,
            labels=stt_backend_labels,
            on_change=_on_stt_backend_change,
        )

        # If local STT is not available, disable the local option in the combobox
        # This is a bit tricky with ttk.Combobox - we'll handle it in the callback
        if not HAS_LOCAL_STT:
            # Store reference to the combobox to disable local option selection
            # The combobox is the last widget in this row
            combobox = frame.grid_slaves(row=row, column=1)[0]
            stt_combobox_ref["widget"] = combobox
        row += 1

        # Helper to create field and track its widgets
        def _create_text_field(label: str, value: str, show: str = "") -> tk.StringVar:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            var = tk.StringVar(master=root, value=value)
            entry = ttk.Entry(frame, textvariable=var, width=48)
            if show:
                entry.configure(show=show)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            # Store label and entry for show/hide
            stt_field_labels[label] = (
                frame.grid_slaves(row=row, column=0)[0],
                frame.grid_slaves(row=row, column=1)[0],
            )
            return var

        def _create_combo_field(label: str, value: str, options: list[str]) -> tk.StringVar:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            var = tk.StringVar(master=root, value=value)
            combo = ttk.Combobox(
                frame, textvariable=var, values=options, width=45, state="readonly"
            )
            combo.grid(row=row, column=1, sticky="ew", pady=2)
            # Store label and combo for show/hide
            stt_field_labels[label] = (
                frame.grid_slaves(row=row, column=0)[0],
                frame.grid_slaves(row=row, column=1)[0],
            )
            return var

        stt_api_url_var = _create_text_field(
            t("settings.stt_api_url"),
            self._config.stt.api_base_url,
        )
        row += 1
        stt_api_key_var = _create_text_field(
            t("settings.stt_api_key"),
            self._config.stt.api_key,
            show="*",
        )
        row += 1
        stt_api_model_var = _create_text_field(
            t("settings.stt_api_model"),
            self._config.stt.api_model,
        )
        row += 1
        stt_model_var = _create_combo_field(
            t("settings.local_model"),
            self._config.stt.model_size,
            ["small", "medium", "large-v3"],
        )
        row += 1
        stt_device_var = _create_combo_field(
            t("settings.local_device"),
            self._config.stt.device,
            ["auto", "cuda", "cpu"],
        )
        row += 1
        # Realtime API fields (Aliyun DashScope)
        stt_realtime_api_key_var = _create_text_field(
            t("settings.stt_realtime_api_key"),
            self._config.stt.realtime_api_key,
            show="*",
        )
        row += 1
        stt_realtime_api_model_var = _create_text_field(
            t("settings.stt_realtime_api_model"),
            self._config.stt.realtime_api_model,
        )
        row += 1

        # Initial show/hide based on current backend
        _on_stt_backend_change(self._config.stt.backend)

        # -- Audio ---------------------------------------------------------
        row = self._heading(frame, t("settings.heading.audio"), row)
        gain_var = self._number_field(
            root,
            frame,
            t("settings.gain_boost"),
            self._config.audio.gain_boost,
            row,
        )
        row += 1

        # -- Overlay -------------------------------------------------------
        row = self._heading(frame, t("settings.heading.overlay"), row)
        capsule_pos_options = ["fixed", "caret"]
        capsule_pos_labels = [
            t("settings.capsule_position.fixed"),
            t("settings.capsule_position.caret"),
        ]
        capsule_pos_var = self._combo_field(
            root,
            frame,
            t("settings.capsule_position"),
            self._config.overlay.capsule_position_mode,
            capsule_pos_options,
            row,
            labels=capsule_pos_labels,
        )
        row += 1

        # -- Language ------------------------------------------------------
        row = self._heading(frame, t("settings.heading.language"), row)
        available_langs = list_available_locales() or ["zh"]
        lang_var = self._combo_field(
            root,
            frame,
            t("settings.language"),
            self._config.language,
            available_langs,
            row,
            labels=[get_locale_display_name(lang) for lang in available_langs],
        )
        row += 1

        # -- Save / Cancel -------------------------------------------------
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=(16, 0), sticky="e")

        ttk.Button(btn_frame, text=t("settings.cancel"), command=root.destroy).pack(
            side="right",
            padx=(8, 0),
        )
        ttk.Button(
            btn_frame,
            text=t("settings.open_logs"),
            command=lambda: self._open_logs_folder(root),
        ).pack(side="right", padx=(8, 0))

        # Rerun wizard button (if callback provided)
        if self._on_rerun_wizard:
            ttk.Button(
                btn_frame,
                text=t("settings.rerun_wizard"),
                command=lambda: self._rerun_wizard(root),
            ).pack(side="right", padx=(8, 0))

        ttk.Button(
            btn_frame,
            text=t("settings.save"),
            command=lambda: self._do_save(
                root,
                hotkey_var=hotkey_var,
                hotkey_mode_var=hotkey_mode_var,
                llm_url_var=llm_url_var,
                llm_key_var=llm_key_var,
                llm_model_var=llm_model_var,
                stt_backend_var=stt_backend_var,
                stt_api_url_var=stt_api_url_var,
                stt_api_key_var=stt_api_key_var,
                stt_api_model_var=stt_api_model_var,
                stt_model_var=stt_model_var,
                stt_device_var=stt_device_var,
                stt_realtime_api_key_var=stt_realtime_api_key_var,
                stt_realtime_api_model_var=stt_realtime_api_model_var,
                gain_var=gain_var,
                capsule_pos_var=capsule_pos_var,
                lang_var=lang_var,
            ),
        ).pack(side="right")

        # Centre the window on screen
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"+{x}+{y}")

        root.mainloop()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _open_logs_folder(self, dialog: tk.Tk) -> None:
        """Open the logs folder in file explorer."""
        log_dir = self._get_log_dir()
        try:
            os.makedirs(log_dir, exist_ok=True)
            # Open folder in default file explorer
            if os.name == "nt":  # Windows
                os.startfile(log_dir)
            elif os.name == "posix":  # macOS and Linux
                if sys.platform == "darwin":
                    subprocess.call(["open", log_dir])
                else:
                    subprocess.call(["xdg-open", log_dir])
            logger.info("Opened logs folder: %s", log_dir)
        except Exception as e:
            logger.exception("Failed to open logs folder: %s", e)

    def _rerun_wizard(self, dialog: tk.Tk) -> None:
        """Rerun the setup wizard."""
        try:
            # Close the settings dialog first
            dialog.destroy()
            # Trigger the wizard rerun callback
            if self._on_rerun_wizard:
                self._on_rerun_wizard()
        except Exception as e:
            logger.exception("Failed to rerun wizard: %s", e)

    @staticmethod
    def _get_log_dir() -> str:
        """Get the log directory path."""
        home = os.path.expanduser("~")
        return os.path.join(home, ".untype", "logs")

    @staticmethod
    def _heading(parent: ttk.Frame, text: str, row: int) -> int:
        """Insert a bold section heading and return the *next* row index."""
        lbl = ttk.Label(parent, text=text, font=("TkDefaultFont", 10, "bold"))
        lbl.grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 4))
        return row + 1

    @staticmethod
    def _text_field(
        master: tk.Tk,
        parent: ttk.Frame,
        label: str,
        value: str,
        row: int,
        *,
        show: str = "",
    ) -> tk.StringVar:
        """Add a labelled text entry and return the associated StringVar."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.StringVar(master=master, value=value)
        entry = ttk.Entry(parent, textvariable=var, width=48)
        if show:
            entry.configure(show=show)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        return var

    @staticmethod
    def _combo_field(
        master: tk.Tk,
        parent: ttk.Frame,
        label: str,
        value: str,
        options: list[str],
        row: int,
        labels: list[str] | None = None,
        on_change: Callable[[str], None] | None = None,
    ) -> tk.StringVar:
        """Add a labelled dropdown and return the associated StringVar.

        Args:
            master: The Tk root.
            parent: Parent frame.
            label: Label text.
            value: Current value (should be one of ``options``).
            options: Internal option values.
            row: Grid row.
            labels: Optional display labels for options (same order as ``options``).
                    If provided, the dropdown shows labels but stores the option value.
            on_change: Optional callback when selection changes, receives new value.
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.StringVar(master=master, value=value)

        if labels is not None:
            # Use labels for display, but we need to handle mapping
            # Create a custom combobox that shows labels but stores values
            # Build a mapping from label to value
            label_to_value = dict(zip(labels, options))
            value_to_label = dict(zip(options, labels))

            # Set the display label as current value
            display_var = tk.StringVar(master=master, value=value_to_label.get(value, value))

            combo = ttk.Combobox(
                parent,
                textvariable=display_var,
                values=labels,
                width=45,
                state="readonly",
            )
            combo.grid(row=row, column=1, sticky="ew", pady=2)

            # When selection changes, update the internal var and call callback
            def _on_select(_event: tk.Event) -> None:  # type: ignore[type-arg]
                selected_label = display_var.get()
                new_value = label_to_value.get(selected_label, selected_label)
                var.set(new_value)
                if on_change:
                    on_change(new_value)

            combo.bind("<<ComboboxSelected>>", _on_select)
        else:
            combo = ttk.Combobox(
                parent, textvariable=var, values=options, width=45, state="readonly"
            )
            combo.grid(row=row, column=1, sticky="ew", pady=2)

            if on_change:

                def _on_select_simple(_event: tk.Event) -> None:  # type: ignore[type-arg]
                    if on_change:
                        on_change(var.get())

                combo.bind("<<ComboboxSelected>>", _on_select_simple)

        return var

    @staticmethod
    def _number_field(
        master: tk.Tk,
        parent: ttk.Frame,
        label: str,
        value: float,
        row: int,
    ) -> tk.DoubleVar:
        """Add a labelled numeric entry and return the associated DoubleVar."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.DoubleVar(master=master, value=value)
        ttk.Entry(parent, textvariable=var, width=48).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        return var

    # Hotkeys that should be blocked because they conflict with system or app functions
    _BLOCKED_HOTKEYS = {
        "ctrl+c",  # Copy - conflicts with app's own clipboard operations
        "ctrl+v",  # Paste
        "ctrl+x",  # Cut
        "ctrl+z",  # Undo
        "ctrl+y",  # Redo
        "ctrl+a",  # Select all
        "ctrl+s",  # Save
        "ctrl+w",  # Close window
        "ctrl+q",  # Quit
        "ctrl+n",  # New
        "ctrl+o",  # Open
        "alt+f4",  # Close app
        "alt+tab",  # Switch window
        "win+l",  # Lock computer
        "escape",  # System key
    }

    @staticmethod
    def _hotkey_field(
        master: tk.Tk,
        parent: ttk.Frame,
        label: str,
        value: str,
        row: int,
    ) -> tk.StringVar:
        """Add a hotkey input field with recording capability.

        Click the input field and press a key (or key combo) to record it.
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.StringVar(master=master, value=value)

        entry = ttk.Entry(parent, textvariable=var, width=48)
        entry.grid(row=row, column=1, sticky="ew", pady=2)

        # Warning label for blocked hotkeys
        warning_var = tk.StringVar(master=master, value="")
        warning_label = ttk.Label(
            parent,
            textvariable=warning_var,
            foreground="red",
            font=("TkDefaultFont", 9),
        )
        warning_label.grid(row=row + 1, column=1, sticky="w", pady=(0, 2))

        # Track recording state
        recording_state = {"active": False}

        def format_key_event(event: tk.Event) -> str:  # type: ignore[type-arg]
            """Convert a key event to a hotkey string."""
            # Build modifiers
            mods = []
            if event.state & 0x0001:  # Shift
                mods.append("shift")
            if event.state & 0x0002:  # Caps Lock (ignore)
                pass
            if event.state & 0x0004:  # Control
                mods.append("ctrl")
            if event.state & 0x0008:  # Alt
                mods.append("alt")

            # Get the key name
            keysym = event.keysym.lower()

            # Normalize key names
            key_map = {
                "control_l": "ctrl",
                "control_r": "ctrl",
                "shift_l": "shift",
                "shift_r": "shift",
                "alt_l": "alt",
                "alt_r": "alt",
                "win_l": "win",
                "win_r": "win",
                "super_l": "win",
                "super_r": "win",
                "space": "space",
                "return": "enter",
                "escape": "esc",
                "backspace": "backspace",
                "tab": "tab",
                "prior": "page_up",
                "next": "page_down",
                "home": "home",
                "end": "end",
                "insert": "insert",
                "delete": "delete",
            }

            # Skip modifier-only presses
            if keysym in (
                "shift_l",
                "shift_r",
                "control_l",
                "control_r",
                "alt_l",
                "alt_r",
                "win_l",
                "win_r",
            ):
                return ""

            key = key_map.get(keysym, keysym)

            # Handle function keys
            if keysym.startswith("f") and keysym[1:].isdigit():
                key = keysym

            # Build final string
            if mods:
                return "+".join(mods + [key])
            return key

        def on_focus_in(_event: tk.Event) -> None:  # type: ignore[type-arg]
            recording_state["active"] = True
            entry.configure(style="Recording.TEntry")

        def on_focus_out(_event: tk.Event) -> None:  # type: ignore[type-arg]
            recording_state["active"] = False
            entry.configure(style="TEntry")

        def on_key_press(event: tk.Event) -> str:  # type: ignore[type-arg]
            if not recording_state["active"]:
                return ""
            hotkey = format_key_event(event)
            if hotkey:
                # Check if hotkey is blocked
                if hotkey.lower() in SettingsDialog._BLOCKED_HOTKEYS:
                    import tkinter.messagebox as messagebox

                    title = (
                        t("settings.error.invalid_hotkey")
                        if hasattr(t, "_dict") and "settings.error.invalid_hotkey" in t._dict
                        else "Invalid Hotkey"
                    )  # noqa: E501
                    msg = (
                        f"{hotkey} cannot be used as it conflicts with "
                        "system or application functions.\n\n"
                        "Please choose a different hotkey (e.g., f6, f7, ctrl+space)."
                    )
                    messagebox.showwarning(title, msg)
                    return "break"
                var.set(hotkey)
                warning_var.set("")
            return "break"  # Prevent default handling

        # Create style for recording state
        style = ttk.Style()
        style.configure("Recording.TEntry", fieldbackground="#e6f3ff")

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
        entry.bind("<KeyPress>", on_key_press)

        # Add hint text
        hint = ttk.Label(
            parent,
            text=t("settings.hotkey_hint"),
            font=("TkDefaultFont", 8),
            foreground="gray",
        )
        hint.grid(row=row + 1, column=1, sticky="w", pady=(0, 2))

        return var

    # ------------------------------------------------------------------ #

    def _do_save(
        self,
        root: tk.Tk,
        *,
        hotkey_var: tk.StringVar,
        hotkey_mode_var: tk.StringVar,
        llm_url_var: tk.StringVar,
        llm_key_var: tk.StringVar,
        llm_model_var: tk.StringVar,
        stt_backend_var: tk.StringVar,
        stt_api_url_var: tk.StringVar,
        stt_api_key_var: tk.StringVar,
        stt_api_model_var: tk.StringVar,
        stt_model_var: tk.StringVar,
        stt_device_var: tk.StringVar,
        stt_realtime_api_key_var: tk.StringVar,
        stt_realtime_api_model_var: tk.StringVar,
        gain_var: tk.DoubleVar,
        capsule_pos_var: tk.StringVar,
        lang_var: tk.StringVar,
    ) -> None:
        """Collect values from the dialog, persist, and notify the app."""
        import copy
        import tkinter.messagebox as messagebox

        try:
            gain_value = gain_var.get()
        except Exception:
            messagebox.showerror("Invalid Gain", t("settings.error.invalid_gain"))
            return

        # Validate gain_boost range
        if not _validate_gain(gain_value):
            messagebox.showerror(
                "Invalid Gain",
                "Gain boost must be between 0.1 and 10.0.",
            )
            return

        # Validate URLs
        llm_url = llm_url_var.get().strip()
        stt_api_url = stt_api_url_var.get().strip()

        if llm_url and not _validate_url(llm_url):
            messagebox.showerror(
                "Invalid URL",
                "LLM base URL must be a valid HTTP/HTTPS URL.",
            )
            return

        if stt_api_url and not _validate_url(stt_api_url):
            messagebox.showerror(
                "Invalid URL",
                "STT API URL must be a valid HTTP/HTTPS URL.",
            )
            return

        # Validate hotkey
        hotkey_value = hotkey_var.get().strip().lower()
        if hotkey_value in SettingsDialog._BLOCKED_HOTKEYS:
            msg = (
                f"{hotkey_value} cannot be used as it conflicts with "
                "system or application functions.\n\n"
                "Please choose a different hotkey (e.g., f6, f7, ctrl+space)."
            )
            messagebox.showwarning("Invalid Hotkey", msg)
            return

        # Create a copy of the config to test save first
        # This prevents in-memory config corruption if save fails
        config_copy = copy.deepcopy(self._config)

        # Apply changes to the copy
        config_copy.hotkey.trigger = hotkey_var.get().strip()
        config_copy.hotkey.mode = hotkey_mode_var.get().strip()
        config_copy.llm.base_url = llm_url_var.get().strip()
        config_copy.llm.api_key = llm_key_var.get().strip()
        config_copy.llm.model = llm_model_var.get().strip()
        config_copy.stt.backend = stt_backend_var.get().strip()
        config_copy.stt.api_base_url = stt_api_url_var.get().strip()
        config_copy.stt.api_key = stt_api_key_var.get().strip()
        config_copy.stt.api_model = stt_api_model_var.get().strip()
        config_copy.stt.model_size = stt_model_var.get().strip()
        config_copy.stt.device = stt_device_var.get().strip()
        config_copy.stt.realtime_api_key = stt_realtime_api_key_var.get().strip()
        config_copy.stt.realtime_api_model = stt_realtime_api_model_var.get().strip()
        config_copy.audio.gain_boost = gain_value
        config_copy.overlay.capsule_position_mode = capsule_pos_var.get().strip()
        config_copy.language = lang_var.get().strip()

        try:
            save_config(config_copy)
            logger.info("Configuration saved")
        except Exception as e:
            logger.exception("Failed to save configuration")
            # Provide more detailed error message
            msg = t("settings.error.save_failed")
            messagebox.showerror(
                "Save Failed", f"{msg}\n\nA backup (.toml.bak) has been preserved.\nError: {e}"
            )
            return

        # Only update in-memory config after successful save
        # Update individual fields to preserve object references
        self._config.hotkey.trigger = config_copy.hotkey.trigger
        self._config.hotkey.mode = config_copy.hotkey.mode
        self._config.llm.base_url = config_copy.llm.base_url
        self._config.llm.api_key = config_copy.llm.api_key
        self._config.llm.model = config_copy.llm.model
        self._config.stt.backend = config_copy.stt.backend
        self._config.stt.api_base_url = config_copy.stt.api_base_url
        self._config.stt.api_key = config_copy.stt.api_key
        self._config.stt.api_model = config_copy.stt.api_model
        self._config.stt.model_size = config_copy.stt.model_size
        self._config.stt.device = config_copy.stt.device
        self._config.stt.realtime_api_key = config_copy.stt.realtime_api_key
        self._config.stt.realtime_api_model = config_copy.stt.realtime_api_model
        self._config.audio.gain_boost = config_copy.audio.gain_boost
        self._config.overlay.capsule_position_mode = config_copy.overlay.capsule_position_mode
        self._config.language = config_copy.language

        root.destroy()

        try:
            self._on_save(self._config)
        except Exception:
            logger.exception("Error in on_save callback")


# ---------------------------------------------------------------------------
# System tray application
# ---------------------------------------------------------------------------


class TrayApp:
    """System tray application manager.

    Provides a pystray-based tray icon with a right-click context menu
    showing the current status, a link to the settings dialog, and a quit
    action.
    """

    def __init__(
        self,
        config: AppConfig,
        on_settings_changed: Callable[[AppConfig], None],
        on_quit: Callable[[], None],
        on_personas_changed: Callable[[], None] | None = None,
        is_recording: Callable[[], bool] | None = None,
        on_rerun_wizard: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._on_settings_changed = on_settings_changed
        self._on_quit = on_quit
        self._on_personas_changed_cb = on_personas_changed
        self._is_recording = is_recording
        self._on_rerun_wizard = on_rerun_wizard
        self._status: str = "Ready"
        self._icon: pystray.Icon | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Run the system tray icon (**blocks** the calling thread)."""
        self._icon = pystray.Icon(
            name="UnType",
            icon=_create_icon_image(_STATUS_COLORS.get(self._status, _DEFAULT_ICON_COLOR)),
            title=f"UnType - {self._status}",
            menu=self._build_menu(),
        )
        logger.info("Starting system tray icon")
        self._icon.run()

    def update_status(self, status: str) -> None:
        """Update the status text and icon colour shown in the tray."""
        self._status = status
        icon = self._icon
        if icon is None:
            return

        color = _STATUS_COLORS.get(status, _DEFAULT_ICON_COLOR)
        icon.icon = _create_icon_image(color)

        # Translate status for display
        status_key = _STATUS_KEYS.get(status, "tray.status.ready")
        translated_status = t(status_key)
        icon.title = f"{t('app.name')} - {translated_status}"

        # Rebuild the menu so the status line reflects the new state.
        icon.menu = self._build_menu()
        icon.update_menu()
        logger.debug("Tray status updated to %r", status)

    def stop(self) -> None:
        """Stop the tray icon and unblock :meth:`run`."""
        icon = self._icon
        if icon is not None:
            icon.stop()
            logger.info("System tray icon stopped")

    # ------------------------------------------------------------------ #
    # Menu construction
    # ------------------------------------------------------------------ #

    def _build_menu(self) -> pystray.Menu:
        """Build the right-click context menu."""
        # Translate status text
        status_key = _STATUS_KEYS.get(self._status, "tray.status.ready")
        translated_status = t(status_key)
        return pystray.Menu(
            pystray.MenuItem(
                f"{t('app.name')} - {translated_status}",
                action=None,
                enabled=False,
            ),
            pystray.MenuItem(
                t("tray.settings"),
                self._on_settings_clicked,
            ),
            pystray.MenuItem(
                t("tray.personas"),
                self._on_personas_clicked,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                t("tray.quit"),
                self._on_quit_clicked,
            ),
        )

    # ------------------------------------------------------------------ #
    # Menu action handlers
    # ------------------------------------------------------------------ #

    def _on_settings_clicked(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Open the settings dialog on a dedicated thread."""
        thread = threading.Thread(
            target=self._show_settings_dialog,
            name="untype-settings-dialog",
            daemon=True,
        )
        thread.start()

    def _on_personas_clicked(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Open the persona manager dialog on a dedicated thread."""
        threading.Thread(
            target=self._show_persona_dialog,
            name="untype-persona-dialog",
            daemon=True,
        ).start()

    def _show_persona_dialog(self) -> None:
        """Create and show the persona manager dialog (runs on its own thread)."""
        try:
            from untype.persona_dialog import PersonaManagerDialog

            dialog = PersonaManagerDialog(on_changed=self._on_personas_changed_cb)
            dialog.show()
        except Exception:
            logger.exception("Failed to open persona manager dialog")

    def _show_settings_dialog(self) -> None:
        """Create and show the settings dialog (runs on its own thread)."""
        # Prevent opening settings during recording to avoid config conflicts
        if self._is_recording and self._is_recording():
            import tkinter.messagebox as messagebox

            # Create a temporary hidden root for the message box
            root = tk.Tk()
            root.withdraw()

            title_key = "settings.error.recording_title"
            msg_key = "settings.error.recording_message"
            title = (
                t(title_key)
                if hasattr(t, "_dict") and title_key in t._dict
                else "Recording in Progress"
            )
            message = (
                t(msg_key)
                if hasattr(t, "_dict") and msg_key in t._dict
                else "Cannot open settings while recording. "
                "Please finish or cancel the recording first."
            )
            messagebox.showwarning(title, message)
            root.destroy()
            return

        try:
            dialog = SettingsDialog(self._config, self._on_settings_saved, self._on_rerun_wizard)
            dialog.show()
        except Exception:
            logger.exception("Failed to open settings dialog")

    def _on_settings_saved(self, config: AppConfig) -> None:
        """Handle a successful save from the settings dialog."""
        self._config = config
        try:
            self._on_settings_changed(config)
        except Exception:
            logger.exception("Error in on_settings_changed callback")

    def _on_quit_clicked(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """Handle the Quit menu item."""
        logger.info("Quit requested via tray menu")
        try:
            self._on_quit()
        except Exception:
            logger.exception("Error in on_quit callback")
        self.stop()
