"""Main entry point — orchestrates all UnType modules together."""

from __future__ import annotations

import copy
import logging
import logging.handlers
import os
import threading
import time
from dataclasses import replace
from datetime import datetime

import numpy as np
import pyperclip

from untype.audio import AudioRecorder, normalize_audio
from untype.build_info import HAS_LOCAL_STT
from untype.clipboard import grab_selected_text, inject_text, release_all_modifiers, save_clipboard
from untype.config import AppConfig, Persona, load_config, load_personas, save_config
from untype.hotkey import HotkeyListener
from untype.i18n import init_language, set_language
from untype.llm import LLMClient
from untype.overlay import CapsuleOverlay
from untype.platform import (
    WindowIdentity,
    get_caret_screen_position,
    get_foreground_window,
    verify_foreground_window,
)
from untype.stt import STTApiEngine, STTEngine, STTRealtimeApiEngine
from untype.tray import TrayApp
from untype.recent_results import RecentResultEntry

logger = logging.getLogger(__name__)

_CONFIG_SAVE_DEBOUNCE_S = 0.25
_CAPTURE_CLEANUP_TIMEOUT_S = 2.0
_REALTIME_STOP_TIMEOUT_S = 3.0
_MAX_RECORDING_PERSONAS = 4
_RECENT_RESULTS_LIMIT = 10


class UnTypeApp:
    """Main application orchestrator.

    Wires together the hotkey listener, audio recorder, STT engine, LLM
    client, and system tray icon into a push-to-talk pipeline.
    """

    def __init__(self) -> None:
        logger.info("Loading configuration...")
        self._config = load_config()
        # Deep copy so settings-change detection works (the dialog mutates in-place)
        self._prev_config = copy.deepcopy(self._config)

        # Initialize i18n with the configured language
        init_language(self._config.language)

        # -- Persona system -----------------------------------------------
        self._personas: list[Persona] = load_personas()
        if self._personas:
            active_count = sum(1 for p in self._personas if p.active)
            logger.info(
                "Loaded %d persona(s) (%d active): %s",
                len(self._personas),
                active_count,
                ", ".join(p.name for p in self._personas if p.active),
            )

        # -- Interaction state (written in on_press, read in pipeline) --------
        self._mode: str = "insert"  # "polish" or "insert"
        self._selected_text: str | None = None
        self._original_clipboard: str | None = None
        self._preselected_persona: Persona | None = None

        # -- HWND safety (Phase 2) --------------------------------------------
        self._target_window: WindowIdentity | None = None
        self._held_result: str | None = None
        self._held_clipboard: str | None = None
        self._held_recent_result_id: str | None = None
        self._window_mismatch: bool = False
        self._hwnd_watch_active: bool = False
        self._caret_x: int = 0
        self._caret_y: int = 0

        # -- Pipeline lock: only one interaction at a time --------------------
        self._pipeline_lock = threading.Lock()
        # True only when _on_hotkey_press succeeded (lock acquired)
        self._press_active = False
        # Signalled once the press-setup thread finishes (recording started or error)
        self._recording_started = threading.Event()
        self._recording_started.set()  # initially "done"
        # Emergency stop: set to abort the pipeline at the next checkpoint.
        self._cancel_requested = threading.Event()

        # -- Recording timeout protection ------------------------------------
        self._timeout_timer: threading.Thread | None = None  # Timer thread for duration check
        self._stop_timeout_timer = threading.Event()  # Signal to stop the timer
        self._stop_timeout_timer.set()  # Initially "stopped"

        # -- Last interaction state (for ghost menu revert/regenerate) ------
        self._last_raw_text: str | None = None
        self._last_result: str | None = None
        self._last_persona: Persona | None = None
        self._last_mode: str | None = None
        self._last_selected_text: str | None = None
        self._last_original_clipboard: str | None = None
        self._last_target_window: WindowIdentity | None = None
        self._last_caret_x: int = 0
        self._last_caret_y: int = 0

        # -- Recent results --------------------------------------------------
        self._recent_results_lock = threading.Lock()
        self._recent_results: list[RecentResultEntry] = []

        # -- Config persistence -----------------------------------------------
        self._config_save_lock = threading.Lock()
        self._config_save_timer: threading.Timer | None = None
        self._config_save_generation = 0

        # -- Initialise subsystems -------------------------------------------
        logger.info("Initialising audio recorder...")
        self._recorder = self._init_recorder()

        logger.info("Initialising system tray...")
        self._tray = TrayApp(
            config=self._config,
            on_settings_changed=self._on_settings_changed,
            on_quit=self._on_quit,
            on_personas_changed=self._on_personas_changed,
            on_rerun_wizard=self._on_rerun_wizard,
            get_recent_results=self._get_recent_results,
            on_recent_copy=self._copy_recent_result,
            on_recent_inject=self._inject_recent_result,
            on_recent_edit=self._edit_recent_result,
        )

        # STT configuration self-check (after tray is ready)
        logger.info("Checking STT configuration...")
        self._handle_stt_config_check()

        logger.info("Initialising STT engine (this may take a moment)...")
        self._stt = self._init_stt()

        logger.info("Initialising LLM client...")
        self._llm: LLMClient | None = self._init_llm_client()

        logger.info("Initialising hotkey listener...")
        self._hotkey = HotkeyListener(
            self._config.hotkey.trigger,
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
            mode=self._config.hotkey.mode,
            on_escape=self._on_cancel,
        )

        logger.info("Initialising overlay...")
        self._overlay = CapsuleOverlay(
            capsule_position_mode=self._config.overlay.capsule_position_mode,
            capsule_fixed_x=self._config.overlay.capsule_fixed_x,
            capsule_fixed_y=self._config.overlay.capsule_fixed_y,
            on_position_changed=self._on_capsule_position_changed,
            on_hold_inject=self._on_hold_inject,
            on_hold_copy=self._on_hold_copy,
            on_hold_ghost=self._on_hold_ghost,
            on_cancel=self._on_cancel,
            on_ghost_revert=self._on_ghost_revert,
            on_ghost_regenerate=self._on_ghost_regenerate,
            on_ghost_use_raw=self._on_ghost_use_raw,
        )

        logger.info("Initialising digit key interceptor...")
        from untype._platform_win32 import DigitKeyInterceptor

        self._digit_interceptor = DigitKeyInterceptor(
            on_digit=self._on_digit_during_recording,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _active_personas(self) -> list[Persona]:
        """Return only active personas (shown during recording)."""
        return [p for p in self._personas if p.active]

    @property
    def _recording_personas(self) -> list[Persona]:
        """Return the smaller recording-time persona subset shown in the quick bar."""
        active = self._active_personas
        quick = [p for p in active if p.quick]
        return (quick or active)[:_MAX_RECORDING_PERSONAS]

    def _start_daemon_thread(self, target, name: str) -> None:
        """Start a daemon thread with consistent naming convention.

        Args:
            target: The callable to run in the thread.
            name: The thread name suffix (will be prefixed with "untype-").
        """
        threading.Thread(
            target=target,
            name=f"untype-{name}",
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the application.

        Starts the hotkey listener, then runs the system-tray icon which
        blocks until the user quits.
        """
        self._hotkey.start()
        self._overlay.start()
        self._digit_interceptor.start()
        mode_hint = (
            "Press %s to start/stop recording."
            if self._config.hotkey.mode == "toggle"
            else "Hold %s to speak."
        )
        logger.info("UnType is ready.  " + mode_hint, self._config.hotkey.trigger)
        self._tray.update_status("Ready")

        # tray.run() blocks until the user selects Quit.
        self._tray.run()

    # ------------------------------------------------------------------
    # Hotkey callbacks
    # ------------------------------------------------------------------

    def _on_hotkey_press(self) -> None:
        """Called when the push-to-talk hotkey is pressed.

        IMPORTANT: This runs inside the pynput low-level keyboard hook
        callback.  On Windows the hook has a strict timeout (~300 ms).
        If we block too long here, Windows removes the hook and the
        hotkey listener dies silently.  Therefore we only acquire the
        pipeline lock and immediately spawn a worker thread for the
        heavy clipboard-probe + recording-start work.
        """
        if not self._pipeline_lock.acquire(blocking=False):
            logger.warning("Pipeline already running — ignoring hotkey press")
            return

        self._cancel_requested.clear()

        # Dismiss any existing ghost menu from a previous interaction.
        self._overlay.hide_ghost_menu()

        # Safety net: if there's a held result from a previous HWND mismatch,
        # copy it to clipboard before discarding so the user doesn't lose it.
        if self._held_result is not None:
            logger.info("Discarding held result to clipboard before new interaction")
            try:
                pyperclip.copy(self._held_result)
            except Exception:
                pass
            self._take_held_result(hide_bubble=True)

        self._press_active = True
        self._recording_started.clear()
        self._start_daemon_thread(self._start_recording, "rec-start")

    def _on_hotkey_release(self) -> None:
        """Called when the push-to-talk hotkey is released.

        Spawns a background thread to run the rest of the pipeline so that
        the hotkey listener is not blocked.

        IMPORTANT: We ALWAYS start a thread to ensure _pipeline_lock gets released,
        even if _press_active was already set to False by _on_cancel (cancel scenario).
        """
        was_active = self._press_active
        self._press_active = False
        # Only start pipeline if this was a genuine hotkey press (not spurious release)
        if was_active:
            self._start_daemon_thread(self._process_pipeline, "pipeline")
        else:
            # Hotkey released but press was already cancelled - start cleanup-only thread
            # to ensure the lock gets released
            self._start_daemon_thread(self._cleanup_after_cancel, "cleanup-after-cancel")

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def _on_cancel(self) -> None:
        """Called from the overlay thread when the user clicks the cancel button."""
        logger.info("Cancel requested by user")
        self._cancel_requested.set()

        # Stop timeout monitor
        self._stop_timeout_monitor()

        # Reset hotkey toggle state so next F6 starts a fresh recording.
        self._hotkey.reset_toggle()

        # Note: Don't call recorder.stop() here - it can deadlock if the audio
        # callback is in progress. Let the pipeline thread handle cleanup.

        self._reset_interaction_ui(hide_hold_bubble=True, cancel_staging=True)

        # CRITICAL: If user cancelled while still holding the hotkey,
        # start cleanup immediately instead of waiting for hotkey release.
        if self._press_active:
            logger.info("Starting immediate cleanup (hotkey still held)")
            self._start_daemon_thread(self._cleanup_after_cancel, "cleanup-after-cancel")

    def _cleanup_after_cancel(self) -> None:
        """Cleanup-only variant of _process_pipeline for cancel scenarios.

        This is called from _on_hotkey_release when the hotkey is released
        but _press_active was already False (meaning _on_cancel was called
        before release). Its only job is to ensure _pipeline_lock is released.

        The real cleanup work is already done by _on_cancel; this just waits
        for _start_recording to finish and then releases the lock.
        """
        try:
            # Wait for _start_recording to finish (up to 5 seconds)
            if not self._recording_started.wait(timeout=5.0):
                logger.warning("Cleanup after cancel: timeout waiting for recording start")

            self._cleanup_active_capture()
        finally:
            # Always release the lock
            self._reset_pipeline_runtime_state()
            self._pipeline_lock.release()
            logger.debug("Cleanup after cancel: lock released")

    # ------------------------------------------------------------------
    # Recording start (runs on a worker thread, NOT the hook thread)
    # ------------------------------------------------------------------

    def _start_recording(self) -> None:
        """Capture HWND, start recording immediately, then probe clipboard.

        Recording starts before the clipboard probe so that the first
        words of speech are not lost to the probe delay.  The mode
        (insert vs polish) is determined after recording is already
        capturing audio.

        This is spawned by :meth:`_on_hotkey_press` on a short-lived worker
        thread so the keyboard hook callback returns immediately.  The
        ``_recording_started`` event is set in the ``finally`` block to let
        the pipeline thread know it can proceed.
        """
        try:
            # Phase 2: capture the target window HWND before anything else.
            self._target_window = get_foreground_window()
            logger.info(
                "Target window: %r (HWND=%d, PID=%d)",
                self._target_window.title,
                self._target_window.hwnd,
                self._target_window.pid,
            )

            # Get caret position for overlay placement.
            caret = get_caret_screen_position()
            self._caret_x = caret.x
            self._caret_y = caret.y
            logger.info(
                "Caret position: (%d, %d) found=%s",
                caret.x,
                caret.y,
                caret.found,
            )

            # For realtime API, establish the WebSocket session FIRST
            # to avoid losing audio data before the connection is ready.
            session_ready = False
            if self._config.stt.backend == "realtime_api":
                if isinstance(self._stt, STTRealtimeApiEngine):
                    logger.info("Starting realtime recognition session...")
                    self._overlay.show(self._caret_x, self._caret_y, "正在连接服务器...")
                    session_ready = self._stt.start_recording_session()
                    if session_ready:
                        logger.info("Realtime recognition session ready")
                    else:
                        logger.warning("Failed to establish realtime recognition session")
                        self._overlay.update_status("连接失败，请检查网络")
                        self._tray.update_status("Error")
                        time.sleep(1.5)
                        self._reset_interaction_ui()
                        return

            # Start recording. For non-realtime backends, this happens first.
            # For realtime, we wait for the session to be ready.
            logger.info("Starting audio recording...")
            self._recorder.start()
            self._tray.update_status("Recording...")

            # Update capsule status to Recording...
            if self._config.stt.backend == "realtime_api" and session_ready:
                # Capsule already shown, just update status
                self._overlay.update_status("Recording...")
            else:
                # Show capsule with Recording status
                self._overlay.show(self._caret_x, self._caret_y, "Recording...")

            # Start the timeout monitor thread
            self._start_timeout_monitor()

            # Show recording persona bar (if personas configured).
            self._preselected_persona = None
            if self._recording_personas:
                persona_tuples = [(p.id, p.icon, p.name) for p in self._recording_personas]
                self._overlay.show_recording_personas(
                    persona_tuples,
                    self._caret_x,
                    self._caret_y,
                    on_click=self._on_rec_persona_click,
                )
                self._digit_interceptor.set_active(True)

                # Auto-select the remembered persona (if any and exists)
                last_id = self._config.last_selected_persona
                if last_id:
                    for idx, p in enumerate(self._recording_personas):
                        if p.id == last_id:
                            self._preselected_persona = p
                            self._overlay.select_recording_persona(idx)
                            logger.info("Auto-selected remembered persona: %s", p.name)
                            break

            # Show realtime preview (after all UI elements are positioned)
            if self._config.stt.backend == "realtime_api" and session_ready:
                self._overlay.show_realtime_preview(self._caret_x, self._caret_y)

            # Grab selected text to determine the interaction mode.
            # This happens while recording is already running.
            logger.info("Grabbing selected text...")
            self._selected_text, self._original_clipboard = grab_selected_text()

            if self._selected_text:
                self._mode = "polish"
                logger.info("Mode: polish (selected %d chars)", len(self._selected_text))
            else:
                self._mode = "insert"
                logger.info("Mode: insert (no selection detected)")

            # Start continuous HWND monitoring.
            self._start_hwnd_watcher()

        except Exception:
            logger.exception("Error starting recording")
            self._tray.update_status("Error")
            self._overlay.update_status("Error")
        finally:
            # Always signal so the pipeline thread never hangs.
            logger.debug("_start_recording: setting _recording_started event")
            self._recording_started.set()
            logger.debug("_start_recording: _recording_started event set")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _process_pipeline(self) -> None:
        """Run the full STT -> staging -> (optional LLM) -> inject pipeline."""
        logger.debug("_process_pipeline: starting, waiting for _recording_started")
        try:
            # Wait for the recording-start thread to finish its work.
            if not self._recording_started.wait(timeout=10.0):
                # Timeout occurred - recording thread crashed or hung
                logger.error("Timeout waiting for recording to start - aborting pipeline")
                self._digit_interceptor.set_active(False)
                self._overlay.hide_recording_personas()
                self._tray.update_status("Error")
                self._overlay.update_status("Timeout")
                self._overlay.hide()
                return

            logger.debug("_process_pipeline: _recording_started is set, checking cancel flag")

            # Checkpoint: cancel requested during recording start?
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled after recording start - initiating cleanup")
                self._cleanup_active_capture()
                self._reset_interaction_ui()
                return

            if not self._recorder.is_recording:
                logger.warning("Recording not active — aborting pipeline")
                self._reset_interaction_ui()
                return

            # Checkpoint: cancel requested before stopping recording?
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled before recording stop - initiating cleanup")
                self._cleanup_active_capture()
                self._reset_interaction_ui()
                return

            # 1. Stop recording and retrieve audio.
            logger.info("Stopping audio recording...")
            self._stop_timeout_monitor()  # Stop timeout monitor before stopping recorder
            audio = self._recorder.stop()

            # Checkpoint: cancel requested during recording stop?
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled after recording stop")
                self._reset_interaction_ui()
                return

            if audio.size == 0:
                logger.warning("Empty audio buffer — aborting pipeline")
                self._reset_interaction_ui()
                return

            # 2. Transcribe (or get realtime result).
            # IMPORTANT: Check cancel before STT processing
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled before STT processing")
                self._cleanup_active_capture()
                self._reset_interaction_ui()
                return

            if self._config.stt.backend == "realtime_api":
                logger.info("Stopping realtime recognition session...")
                text = self._get_realtime_result()
                text = text.strip()
            else:
                # For API and local backends, normalize and transcribe.
                self._tray.update_status("Transcribing...")
                self._overlay.update_status("Transcribing...")
                logger.info("Normalising audio (gain=%.1f)...", self._config.audio.gain_boost)
                audio = normalize_audio(audio, self._config.audio.gain_boost)

                logger.info("Transcribing audio...")
                text = self._stt.transcribe(audio)
                text = text.strip()

            # Checkpoint: cancel requested during transcription?
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled after transcription")
                self._reset_interaction_ui()
                return

            if not text:
                logger.warning("Empty transcription — aborting pipeline")
                self._reset_interaction_ui()
                return

            logger.info("Transcription: %s", text)

            # 4. STT complete — deactivate digit interceptor and hide persona bar.
            self._digit_interceptor.set_active(False)
            self._overlay.hide_recording_personas()

            # 5. Branch: personas configured → skip staging; otherwise → staging.
            if self._active_personas:
                self._process_with_personas(text)
            else:
                self._process_with_staging(text)

        except Exception:
            logger.exception("Pipeline error")
            self._hwnd_watch_active = False
            self._digit_interceptor.set_active(False)
            self._tray.update_status("Error")
            self._overlay.update_status("Error")
            # Briefly show the error status, then revert to Ready.
            time.sleep(2)
            self._tray.update_status("Ready")
            self._overlay.hide()
            self._overlay.hide_recording_personas()
        finally:
            self._hwnd_watch_active = False
            self._digit_interceptor.set_active(False)
            self._cancel_requested.clear()
            self._pipeline_lock.release()

    def _run_llm(self, transcribed_text: str, persona: Persona | None = None) -> str:
        """Send transcribed text through the LLM, falling back to raw text.

        If *persona* is provided, its prompt/model/temperature/max_tokens
        overrides are passed to the LLM client for this single call.

        The LLM call can be cancelled by setting _cancel_requested event.
        """
        if self._llm is None:
            logger.warning("LLM not configured — using raw transcription")
            return transcribed_text

        # Build per-call overrides from persona.
        overrides: dict = {}
        if persona:
            if self._mode == "polish" and persona.prompt_polish:
                overrides["system_prompt"] = persona.prompt_polish
            elif self._mode == "insert" and persona.prompt_insert:
                overrides["system_prompt"] = persona.prompt_insert
            if persona.model:
                overrides["model"] = persona.model
            if persona.temperature is not None:
                overrides["temperature"] = persona.temperature
            if persona.max_tokens is not None:
                overrides["max_tokens"] = persona.max_tokens

        # Pass cancel_event to enable interruption
        overrides["cancel_event"] = self._cancel_requested

        try:
            if self._mode == "polish":
                logger.info("Polishing selected text with LLM...")
                return self._llm.polish(
                    self._selected_text or "",
                    transcribed_text,
                    **overrides,
                )
            else:
                logger.info("Inserting text via LLM...")
                return self._llm.insert(transcribed_text, **overrides)
        except KeyboardInterrupt:
            logger.info("LLM request cancelled by user")
            raise
        except Exception:
            logger.exception("LLM request failed — falling back to raw transcription")
            return transcribed_text

    def _show_processing_ui(self, *, at_corner: bool = False) -> None:
        """Show the processing capsule in its normal or parked state."""
        self._hwnd_watch_active = False
        if at_corner:
            self._overlay.update_status("Processing...")
        else:
            self._overlay.show(self._caret_x, self._caret_y, "Processing...")
        self._tray.update_status("Processing...")

    def _run_llm_with_watch(
        self,
        llm_input: str,
        *,
        persona: Persona | None = None,
        at_corner: bool = False,
        cancel_before_message: str,
        cancel_during_message: str,
        cancel_after_message: str,
        error_message: str,
    ) -> str | None:
        """Run an LLM request with shared processing UI and cancellation handling."""
        self._show_processing_ui(at_corner=at_corner)
        self._start_hwnd_watcher()

        if self._cancel_requested.is_set():
            logger.info(cancel_before_message)
            self._reset_interaction_ui()
            return None

        try:
            result = self._run_llm(llm_input, persona=persona)
        except KeyboardInterrupt:
            logger.info(cancel_during_message)
            self._reset_interaction_ui()
            return None
        except Exception:
            logger.exception(error_message)
            self._reset_interaction_ui()
            return None

        if self._cancel_requested.is_set():
            logger.info(cancel_after_message)
            self._reset_interaction_ui()
            return None

        self._hwnd_watch_active = False
        return result

    def _process_with_personas(self, text: str) -> None:
        """Fast-lane: skip staging, go directly to LLM (with optional persona).

        Called when personas are configured.  The user may have pre-selected
        a persona via digit keys during recording/STT.
        """
        persona = self._preselected_persona
        self._preselected_persona = None

        if persona:
            logger.info("Using pre-selected persona: %s", persona.name)
        else:
            logger.info("No persona pre-selected — using default LLM processing")

        self._hwnd_watch_active = False
        at_corner = self._window_mismatch

        result = self._run_llm_with_watch(
            text,
            persona=persona,
            at_corner=at_corner,
            cancel_before_message="Pipeline cancelled before LLM call (fast-lane)",
            cancel_during_message="Pipeline cancelled during LLM (fast-lane)",
            cancel_after_message="Pipeline cancelled after LLM returned (fast-lane)",
            error_message="Pipeline error in _process_with_personas",
        )
        if result is None:
            return

        # Stop watcher before the unified delivery step.
        self._hwnd_watch_active = False
        logger.info("Injecting refined text (%d chars)", len(result))
        if self._deliver_result(text, result, persona=persona):
            logger.info("Pipeline complete (fast-lane)")

    def _process_with_staging(self, text: str) -> None:
        """Show staging area for manual editing (no personas configured)."""
        self._hwnd_watch_active = False
        at_corner = self._window_mismatch
        self._tray.update_status("Ready")

        if at_corner:
            self._overlay.show_staging(text, 0, 0, at_corner=True)
        else:
            self._overlay.show_staging(
                text,
                self._caret_x,
                self._caret_y,
            )

        # Block until the user acts (Enter / Shift+Enter / Esc).
        edited_text, action = self._overlay.wait_staging()

        if action == "cancel":
            logger.info("Staging cancelled by user")
            return

        # Checkpoint: cancel requested while staging was open?
        if self._cancel_requested.is_set():
            logger.info("Pipeline cancelled during staging")
            return

        # Brief delay for focus to return to the previous window.
        time.sleep(0.15)

        if action == "raw":
            logger.info("Injecting raw text (%d chars)", len(edited_text))
            self._deliver_result(text, edited_text)
            return

        # action == "refine" — send through LLM.
        result = self._run_llm_with_watch(
            edited_text,
            cancel_before_message="Pipeline cancelled before LLM call (staging)",
            cancel_during_message="Pipeline cancelled during LLM (staging)",
            cancel_after_message="Pipeline cancelled after LLM returned (staging)",
            error_message="Pipeline error in _process_with_staging LLM",
        )
        if result is None:
            return

        # Stop watcher before the unified delivery step.
        self._hwnd_watch_active = False
        logger.info("Injecting refined text (%d chars)", len(result))
        if self._deliver_result(text, result):
            logger.info("Pipeline complete")

    # ------------------------------------------------------------------
    # HWND watcher (Phase 2 — polls foreground window during pipeline)
    # ------------------------------------------------------------------

    def _start_hwnd_watcher(self) -> None:
        """Begin polling the foreground window on a daemon thread."""
        self._window_mismatch = False
        self._hwnd_watch_active = True
        self._start_daemon_thread(self._watch_hwnd, "hwnd-watch")

    def _watch_hwnd(self) -> None:
        """Poll foreground window every 200ms.  Triggers capsule flight on mismatch."""
        try:
            while self._hwnd_watch_active:
                time.sleep(0.2)
                if not self._hwnd_watch_active:
                    break
                if self._target_window is not None and not verify_foreground_window(
                    self._target_window
                ):
                    self._window_mismatch = True
                    logger.info(
                        "Window switch detected during pipeline (expected HWND=%d, title=%r)",
                        self._target_window.hwnd,
                        self._target_window.title,
                    )
                    self._overlay.fly_to_corner()
                    break
        except Exception:
            # Log any unexpected error and exit cleanly rather than hanging
            logger.exception("HWND watcher thread encountered an error")
            self._window_mismatch = True
            try:
                self._overlay.fly_to_corner()
            except Exception:
                logger.exception("Failed to fly capsule to corner during HWND watcher error")
        finally:
            # Ensure flag is cleared even if we exited abnormally
            self._hwnd_watch_active = False

    def _verify_window_safety(self) -> bool:
        """Check if the target window is still safe for text injection.

        Returns False if the target window is set and either:
        - A window mismatch was previously detected, or
        - The foreground window no longer matches the target.
        """
        if self._target_window is None:
            return True
        return not (self._window_mismatch or not verify_foreground_window(self._target_window))

    def _reset_interaction_ui(
        self,
        *,
        tray_status: str = "Ready",
        hide_hold_bubble: bool = False,
        cancel_staging: bool = False,
    ) -> None:
        """Reset transient UI used by an in-flight interaction."""
        self._hwnd_watch_active = False
        self._digit_interceptor.set_active(False)
        self._overlay.hide()
        self._overlay.hide_recording_personas()
        self._overlay.hide_realtime_preview()
        if hide_hold_bubble:
            self._overlay.hide_hold_bubble()
        if cancel_staging:
            self._overlay.cancel_staging()
        self._tray.update_status(tray_status)

    def _reset_pipeline_runtime_state(self) -> None:
        """Clear transient pipeline flags after cleanup or completion."""
        self._hwnd_watch_active = False
        self._digit_interceptor.set_active(False)
        self._cancel_requested.clear()

    def _save_config_snapshot(self, config: AppConfig, *, reason: str) -> None:
        """Persist a config snapshot without mutating in-flight state."""
        try:
            save_config(config)
            logger.debug("Saved configuration snapshot (%s)", reason)
        except Exception:
            logger.exception("Failed to save configuration snapshot (%s)", reason)

    def _run_scheduled_config_save(
        self,
        generation: int,
        snapshot: AppConfig,
        *,
        reason: str,
    ) -> None:
        """Persist a debounced config snapshot if it is still current."""
        with self._config_save_lock:
            if generation != self._config_save_generation:
                return
            self._config_save_timer = None
        self._save_config_snapshot(snapshot, reason=reason)

    def _schedule_config_save(self, *, delay: float, reason: str) -> None:
        """Debounce config saves for high-frequency state changes."""
        snapshot = copy.deepcopy(self._config)
        with self._config_save_lock:
            self._config_save_generation += 1
            generation = self._config_save_generation
            if self._config_save_timer is not None:
                self._config_save_timer.cancel()
            timer = threading.Timer(
                delay,
                self._run_scheduled_config_save,
                args=(generation, snapshot),
                kwargs={"reason": reason},
            )
            timer.name = f"untype-config-save-{generation}"
            timer.daemon = True
            self._config_save_timer = timer
            timer.start()

    def _save_config_now(self, *, reason: str) -> None:
        """Cancel pending debounced writes and save the latest config immediately."""
        snapshot = copy.deepcopy(self._config)
        with self._config_save_lock:
            self._config_save_generation += 1
            if self._config_save_timer is not None:
                self._config_save_timer.cancel()
                self._config_save_timer = None
        self._save_config_snapshot(snapshot, reason=reason)

    def _flush_pending_config_save(self) -> None:
        """Persist any pending debounced config update before shutdown."""
        with self._config_save_lock:
            if self._config_save_timer is None:
                return
            self._config_save_generation += 1
            self._config_save_timer.cancel()
            self._config_save_timer = None
            snapshot = copy.deepcopy(self._config)
        self._save_config_snapshot(snapshot, reason="shutdown flush")

    def _call_with_timeout(
        self,
        func,
        timeout: float,
        *,
        description: str,
        fallback=None,
    ):
        """Run *func* on a worker thread with a bounded timeout."""
        import concurrent.futures

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(func)
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning("%s timed out", description)
        except Exception as exc:
            logger.warning("%s failed: %s", description, exc)
        finally:
            executor.shutdown(wait=False)

        if fallback is not None:
            return fallback()
        return None

    def _stop_realtime_session(self) -> str:
        """Stop the realtime STT session and return the accumulated text."""
        if isinstance(self._stt, STTRealtimeApiEngine):
            return self._stt.stop_session()
        return ""

    def _cleanup_capture_resources(self) -> None:
        """Abort active recording and stop any live realtime session."""
        if self._recorder.is_recording:
            logger.debug("Cleanup: aborting recorder")
            self._recorder.abort()
        if isinstance(self._stt, STTRealtimeApiEngine):
            logger.debug("Cleanup: stopping STT session")
            self._stt.stop_session()

    def _cleanup_active_capture(self) -> None:
        """Best-effort cleanup for in-flight recording/STT resources."""
        self._call_with_timeout(
            self._cleanup_capture_resources,
            _CAPTURE_CLEANUP_TIMEOUT_S,
            description="Capture cleanup",
        )

    def _get_realtime_result(self) -> str:
        """Stop realtime STT with timeout and fall back to the current partial text."""
        result = self._call_with_timeout(
            self._stop_realtime_session,
            _REALTIME_STOP_TIMEOUT_S,
            description="STT stop_session",
            fallback=lambda: self._stt.get_result()
            if isinstance(self._stt, STTRealtimeApiEngine)
            else "",
        )
        return result or ""

    def _hold_result_for_later(
        self,
        raw_text: str,
        result: str,
        *,
        persona: Persona | None = None,
        recent_result_id: str | None = None,
        record_recent: bool = True,
    ) -> None:
        """Persist *result* for the hold bubble instead of injecting it now."""
        self._save_interaction_state(raw_text, result, persona=persona, show_ghost=False)
        self._held_result = result
        self._held_clipboard = self._original_clipboard
        if recent_result_id is not None:
            self._held_recent_result_id = recent_result_id
            self._update_recent_result_status(recent_result_id, "held")
        elif record_recent:
            self._held_recent_result_id = self._record_recent_result(
                raw_text,
                result,
                persona=persona,
                status="held",
            )
        else:
            self._held_recent_result_id = None
        self._overlay.fly_to_hold_bubble(result)
        self._tray.update_status("Ready")

    def _deliver_result(
        self,
        raw_text: str,
        result: str,
        *,
        persona: Persona | None = None,
        recent_result_id: str | None = None,
        record_recent: bool = True,
    ) -> bool:
        """Inject *result* or divert it to the hold bubble when unsafe."""
        if not self._verify_window_safety():
            logger.warning("Window changed before injection - holding result")
            self._hold_result_for_later(
                raw_text,
                result,
                persona=persona,
                recent_result_id=recent_result_id,
                record_recent=record_recent,
            )
            return False

        injection = inject_text(result, self._original_clipboard)
        if not injection:
            logger.warning(
                "Text injection failed (method=%s, typed=%d, clipboard=%s, paste=%s) - holding result",
                injection.delivery_method,
                injection.typed_characters,
                injection.copied_to_clipboard,
                injection.paste_simulated,
            )
            self._hold_result_for_later(
                raw_text,
                result,
                persona=persona,
                recent_result_id=recent_result_id,
                record_recent=record_recent,
            )
            return False

        if recent_result_id is not None:
            self._update_recent_result_status(recent_result_id, "injected")
        elif record_recent:
            self._record_recent_result(
                raw_text,
                result,
                persona=persona,
                status="injected",
            )

        self._save_interaction_state(raw_text, result, persona=persona)
        self._overlay.hide()
        self._tray.update_status("Ready")
        return True

    # ------------------------------------------------------------------
    # Hold callbacks (Phase 2 — called from overlay thread)
    # ------------------------------------------------------------------

    def _take_held_result(
        self,
        *,
        hide_bubble: bool = False,
    ) -> tuple[str | None, str | None, str | None]:
        """Consume and clear the currently held result."""
        result = self._held_result
        clipboard = self._held_clipboard
        recent_result_id = self._held_recent_result_id
        self._held_result = None
        self._held_clipboard = None
        self._held_recent_result_id = None
        if hide_bubble:
            self._overlay.hide_hold_bubble()
        return result, clipboard, recent_result_id

    def _restore_held_result(
        self,
        result: str | None,
        clipboard: str | None,
        recent_result_id: str | None = None,
        *,
        show_bubble: bool = False,
    ) -> None:
        """Restore a held result after a failed retry path."""
        self._held_result = result
        self._held_clipboard = clipboard
        self._held_recent_result_id = recent_result_id
        if show_bubble and result is not None:
            self._overlay.fly_to_hold_bubble(result)

    def _try_acquire_pipeline_action(self, action_name: str) -> bool:
        """Acquire the pipeline lock for a ghost action or log a consistent warning."""
        if not self._pipeline_lock.acquire(blocking=False):
            logger.warning("%s: pipeline busy — ignoring", action_name)
            return False
        return True

    def _on_hold_inject(self) -> None:
        """Left-click on hold bubble — inject into the current foreground window."""
        result, clipboard, recent_result_id = self._take_held_result()

        if result is None:
            return

        logger.info("Hold-inject: injecting %d chars into current window", len(result))
        injection = inject_text(result, clipboard)
        if not injection:
            logger.warning(
                "Hold-inject failed (method=%s, typed=%d, clipboard=%s, paste=%s) - restoring hold bubble",
                injection.delivery_method,
                injection.typed_characters,
                injection.copied_to_clipboard,
                injection.paste_simulated,
            )
            self._restore_held_result(
                result,
                clipboard,
                recent_result_id,
                show_bubble=True,
            )
            return

        if recent_result_id is not None:
            self._update_recent_result_status(recent_result_id, "injected")

        # Show ghost menu if we have saved interaction state.
        if self._last_raw_text is not None:
            self._last_result = result
            self._set_last_delivery_anchor(get_foreground_window(), show_ghost=True)

    def _on_hold_copy(self) -> None:
        """Right-click on hold bubble — copy held text to clipboard."""
        result, _clipboard, recent_result_id = self._take_held_result()

        if result is None:
            return

        logger.info("Hold-copy: copying %d chars to clipboard", len(result))
        try:
            pyperclip.copy(result)
            if recent_result_id is not None:
                self._update_recent_result_status(recent_result_id, "copied")
        except Exception:
            logger.exception("Failed to copy held result to clipboard")

    def _on_hold_ghost(self) -> None:
        """Middle-click on hold bubble — show ghost menu for revert/regenerate.

        Does NOT inject text.  The held result is discarded (the ghost
        menu actions will use ``_last_*`` state instead).
        """
        self._take_held_result()

        if self._last_raw_text is None:
            logger.warning("Hold-ghost: no interaction state saved")
            return

        logger.info("Hold-ghost: showing ghost menu")
        caret = get_caret_screen_position()
        self._overlay.show_ghost_menu(caret.x, caret.y)

    # ------------------------------------------------------------------
    # Ghost menu callbacks (post-injection revert/regenerate)
    # ------------------------------------------------------------------

    def _save_interaction_state(
        self,
        raw_text: str,
        result: str,
        persona: Persona | None = None,
        show_ghost: bool = True,
    ) -> None:
        """Capture the current interaction state for ghost menu revert/regenerate.

        When *show_ghost* is True (default), the ghost menu icon is shown
        near the caret.  Set to False when diverting to the hold bubble
        (ghost will be shown after hold-inject instead).
        """
        self._last_raw_text = raw_text
        self._last_result = result
        self._last_persona = persona
        self._last_mode = self._mode
        self._last_selected_text = self._selected_text
        self._last_original_clipboard = self._original_clipboard
        self._set_last_delivery_anchor(self._target_window, show_ghost=show_ghost)

    def _set_last_delivery_anchor(
        self,
        target_window: WindowIdentity | None,
        *,
        show_ghost: bool,
    ) -> None:
        """Update the saved caret/window anchor for ghost actions."""
        self._last_target_window = target_window
        caret = get_caret_screen_position()
        self._last_caret_x = caret.x
        self._last_caret_y = caret.y
        if show_ghost:
            self._overlay.show_ghost_menu(caret.x, caret.y)

    def _restore_last_interaction_context(self, target_window: WindowIdentity | None) -> None:
        """Restore saved interaction context before a ghost action reinjects text."""
        self._mode = self._last_mode or "insert"
        self._selected_text = self._last_selected_text
        self._original_clipboard = self._last_original_clipboard
        self._target_window = target_window
        self._caret_x = self._last_caret_x
        self._caret_y = self._last_caret_y
        self._window_mismatch = False

    def _undo_last_injection_if_possible(self, action_name: str) -> WindowIdentity | None:
        """Undo the last injected result when the original target is still foreground."""
        target = self._last_target_window
        if target is not None and verify_foreground_window(target):
            logger.info("%s: undoing paste via Ctrl+Z", action_name)
            self._simulate_undo()
        else:
            logger.info("%s: target window not in foreground, skipping undo", action_name)
        return target

    def _get_active_persona_staging_options(self) -> list[tuple[str, str, str]] | None:
        """Build staging persona options for the overlay when personas are active."""
        if not self._active_personas:
            return None
        return [(p.id, p.icon, p.name) for p in self._active_personas]

    def _record_recent_result(
        self,
        raw_text: str,
        result: str,
        *,
        persona: Persona | None,
        status: str,
    ) -> str:
        """Add a new recent result entry and keep the session history bounded."""
        target = self._target_window
        entry = RecentResultEntry(
            id=f"recent-{time.time_ns()}",
            created_at=datetime.now(),
            raw_text=raw_text,
            result_text=result,
            mode=self._mode,
            status=status,
            window_title=(target.title.strip() if target is not None else ""),
            persona_id=persona.id if persona is not None else None,
            persona_name=persona.name if persona is not None else None,
            persona_icon=persona.icon if persona is not None else None,
        )
        with self._recent_results_lock:
            self._recent_results.insert(0, entry)
            del self._recent_results[_RECENT_RESULTS_LIMIT:]
        return entry.id

    def _update_recent_result_status(self, entry_id: str | None, status: str) -> None:
        """Update the delivery status for an existing recent-result entry."""
        if entry_id is None:
            return
        with self._recent_results_lock:
            for index, entry in enumerate(self._recent_results):
                if entry.id == entry_id:
                    self._recent_results[index] = replace(entry, status=status)
                    return

    def _get_recent_results(self) -> list[RecentResultEntry]:
        """Return a snapshot of recent results for the tray dialog."""
        with self._recent_results_lock:
            return list(self._recent_results)

    def _find_recent_result(self, entry_id: str) -> RecentResultEntry | None:
        """Look up a recent result by id."""
        with self._recent_results_lock:
            for entry in self._recent_results:
                if entry.id == entry_id:
                    return entry
        return None

    def _prepare_foreground_delivery_context(self, *, mode: str = "insert") -> None:
        """Capture the current foreground target for a manual delivery action."""
        self._mode = mode
        self._selected_text = None
        self._original_clipboard = save_clipboard()
        time.sleep(0.05)
        target = get_foreground_window()
        self._target_window = target if target.hwnd else None
        caret = get_caret_screen_position()
        self._caret_x = caret.x
        self._caret_y = caret.y
        self._window_mismatch = False

    def _copy_recent_result(self, entry_id: str) -> bool:
        """Copy a recent result back to the clipboard."""
        entry = self._find_recent_result(entry_id)
        if entry is None:
            return False

        try:
            pyperclip.copy(entry.result_text)
        except Exception:
            logger.exception("Recent result copy failed")
            return False

        if entry.status == "held":
            self._update_recent_result_status(entry_id, "copied")
        return True

    def _inject_recent_result(self, entry_id: str) -> bool:
        """Reinject a recent result into the current foreground window."""
        entry = self._find_recent_result(entry_id)
        if entry is None:
            return False
        if not self._try_acquire_pipeline_action("Recent inject"):
            return False

        try:
            self._prepare_foreground_delivery_context()
            return self._deliver_result(
                entry.result_text,
                entry.result_text,
                recent_result_id=entry_id,
                record_recent=False,
            )
        finally:
            self._reset_pipeline_runtime_state()

    def _edit_recent_result(self, entry_id: str) -> bool:
        """Open a recent result in staging for manual revision and reinjection."""
        entry = self._find_recent_result(entry_id)
        if entry is None:
            return False
        if not self._try_acquire_pipeline_action("Recent edit"):
            return False

        try:
            self._prepare_foreground_delivery_context()
            self._tray.update_status("Ready")
            self._overlay.show_staging(
                entry.result_text,
                self._caret_x,
                self._caret_y,
                personas=self._get_active_persona_staging_options(),
            )
            edited_text, action = self._overlay.wait_staging()

            if action == "cancel":
                logger.info("Recent edit staging: cancelled")
                return True

            time.sleep(0.15)

            persona = None
            if action.startswith("persona:"):
                pid = action.split(":", 1)[1]
                persona = next((p for p in self._personas if p.id == pid), None)
                action = "refine"

            if action == "raw":
                return self._deliver_result(entry.result_text, edited_text, persona=persona)

            result = self._run_llm_with_watch(
                edited_text,
                persona=persona,
                cancel_before_message="Recent edit: cancelled before LLM call",
                cancel_during_message="Recent edit: LLM cancelled by user",
                cancel_after_message="Recent edit: cancelled after LLM returned",
                error_message="Recent edit LLM error",
            )
            if result is None:
                return True

            self._hwnd_watch_active = False
            return self._deliver_result(entry.result_text, result, persona=persona)
        finally:
            self._reset_pipeline_runtime_state()
            self._tray.update_status("Ready")

    def _simulate_undo(self) -> None:
        """Send Ctrl+Z to undo the last paste in the target app."""
        from pynput.keyboard import Controller, Key

        kbd = Controller()
        release_all_modifiers()
        time.sleep(0.05)
        kbd.press(Key.ctrl_l)
        time.sleep(0.05)
        kbd.press("z")
        time.sleep(0.02)
        kbd.release("z")
        time.sleep(0.02)
        kbd.release(Key.ctrl_l)
        time.sleep(0.1)

    def _on_ghost_revert(self) -> None:
        """Ghost menu 'Revert' — undo paste, reopen staging with raw text."""
        raw_text = self._last_raw_text
        if raw_text is None:
            logger.warning("Ghost revert: no interaction state saved")
            return

        if not self._try_acquire_pipeline_action("Ghost revert"):
            return

        try:
            target = self._undo_last_injection_if_possible("Ghost revert")
            self._restore_last_interaction_context(target)

            # Clear last state before re-entering staging.
            self._last_raw_text = None
            self._last_result = None

            # Show staging area with the raw text.
            self._overlay.show_staging(
                raw_text,
                self._caret_x,
                self._caret_y,
                personas=self._get_active_persona_staging_options(),
            )

            # Block until user acts.
            edited_text, action = self._overlay.wait_staging()

            if action == "cancel":
                logger.info("Ghost revert staging: cancelled")
                return

            time.sleep(0.15)

            persona = None
            if action.startswith("persona:"):
                pid = action.split(":", 1)[1]
                persona = next(
                    (p for p in self._personas if p.id == pid),
                    None,
                )
                action = "refine"

            if action == "raw":
                logger.info("Ghost revert: injecting raw text (%d chars)", len(edited_text))
                self._deliver_result(raw_text, edited_text)
                return

            result = self._run_llm_with_watch(
                edited_text,
                persona=persona,
                cancel_before_message="Ghost revert: cancelled before LLM call",
                cancel_during_message="Ghost revert: LLM cancelled by user",
                cancel_after_message="Ghost revert: cancelled after LLM returned",
                error_message="Ghost revert LLM error",
            )
            if result is None:
                return

            # Stop watcher before the unified delivery step.
            self._hwnd_watch_active = False
            logger.info("Ghost revert: injecting refined text (%d chars)", len(result))
            self._deliver_result(raw_text, result, persona=persona)

        except Exception:
            logger.exception("Ghost revert error")
            self._reset_interaction_ui()
        finally:
            self._hwnd_watch_active = False
            self._pipeline_lock.release()

    def _on_ghost_regenerate(self) -> None:
        """Ghost menu 'Regenerate' — undo paste, re-run LLM, inject new result."""
        raw_text = self._last_raw_text
        persona = self._last_persona
        if raw_text is None:
            logger.warning("Ghost regenerate: no interaction state saved")
            return

        if not self._try_acquire_pipeline_action("Ghost regenerate"):
            return

        try:
            target = self._undo_last_injection_if_possible("Ghost regenerate")
            self._restore_last_interaction_context(target)

            result = self._run_llm_with_watch(
                raw_text,
                persona=persona,
                cancel_before_message="Ghost regenerate: cancelled before LLM call",
                cancel_during_message="Ghost regenerate: LLM cancelled by user",
                cancel_after_message="Ghost regenerate: cancelled after LLM returned",
                error_message="Ghost regenerate LLM error",
            )
            if result is None:
                return

            # Stop watcher before the unified delivery step.
            self._hwnd_watch_active = False
            logger.info("Ghost regenerate: injecting refined text (%d chars)", len(result))
            self._deliver_result(raw_text, result, persona=persona)

        except Exception:
            logger.exception("Ghost regenerate error")
            self._reset_interaction_ui()
        finally:
            self._hwnd_watch_active = False
            self._pipeline_lock.release()

    def _on_ghost_use_raw(self) -> None:
        """Ghost menu 'Use Raw' — undo paste, inject raw STT text instead."""
        raw_text = self._last_raw_text
        if raw_text is None:
            logger.warning("Ghost use-raw: no interaction state saved")
            return

        if not self._try_acquire_pipeline_action("Ghost use-raw"):
            return

        try:
            target = self._undo_last_injection_if_possible("Ghost use-raw")
            self._restore_last_interaction_context(target)

            logger.info("Ghost use-raw: injecting raw text (%d chars)", len(raw_text))
            self._deliver_result(raw_text, raw_text)

        except Exception:
            logger.exception("Ghost use-raw error")
        finally:
            self._pipeline_lock.release()

    # ------------------------------------------------------------------
    # Recording persona callbacks
    # ------------------------------------------------------------------

    def _on_digit_during_recording(self, digit: int) -> None:
        """Called from the keyboard hook thread when a digit key is pressed."""
        idx = digit - 1
        if idx < len(self._recording_personas):
            active_persona = self._recording_personas[idx]
            if self._preselected_persona == active_persona:
                # Toggle off: pressing the same digit again deselects.
                self._preselected_persona = None
                self._overlay.select_recording_persona(-1)
                self._save_selected_persona(None)
            else:
                self._preselected_persona = active_persona
                self._overlay.select_recording_persona(idx)
                self._save_selected_persona(active_persona.id)

    def _save_selected_persona(self, persona_id: str | None) -> None:
        """Save the selected persona ID to config."""
        # Save "default" as empty string (the actual default)
        if persona_id == "default":
            persona_id = None

        self._config.last_selected_persona = persona_id or "default"
        self._schedule_config_save(
            delay=_CONFIG_SAVE_DEBOUNCE_S,
            reason="persona selection",
        )

    def _on_rec_persona_click(self, index: int) -> None:
        """Called from overlay thread when a recording persona button is clicked."""
        self._on_digit_during_recording(index + 1)

    # ------------------------------------------------------------------
    # Settings hot-reload
    # ------------------------------------------------------------------

    def _on_settings_changed(self, new_config: AppConfig) -> None:
        """Handle settings changes pushed from the tray settings dialog."""
        old = self._prev_config
        self._config = new_config
        self._prev_config = copy.deepcopy(new_config)

        logger.info("Saving updated configuration...")
        self._save_config_now(reason="settings update")

        # --- Language ---
        if new_config.language != old.language:
            logger.info("Language changed (%r→%r)", old.language, new_config.language)
            set_language(new_config.language)

        # --- Hotkey ---
        if (
            new_config.hotkey.trigger != old.hotkey.trigger
            or new_config.hotkey.mode != old.hotkey.mode
        ):
            logger.info(
                "Hotkey changed (trigger=%r→%r, mode=%r→%r) — restarting listener",
                old.hotkey.trigger,
                new_config.hotkey.trigger,
                old.hotkey.mode,
                new_config.hotkey.mode,
            )
            self._hotkey.stop()
            self._hotkey = HotkeyListener(
                new_config.hotkey.trigger,
                on_press=self._on_hotkey_press,
                on_release=self._on_hotkey_release,
                mode=new_config.hotkey.mode,
                on_escape=self._on_cancel,
            )
            self._hotkey.start()

        # --- Audio recorder ---
        if (
            new_config.audio.sample_rate != old.audio.sample_rate
            or new_config.audio.device != old.audio.device
        ):
            logger.info("Audio settings changed — reinitialising recorder")
            self._recorder = self._init_recorder()

        # --- STT engine ---
        stt_changed = (
            new_config.stt.backend != old.stt.backend
            or new_config.stt.model_size != old.stt.model_size
            or new_config.stt.device != old.stt.device
            or new_config.stt.compute_type != old.stt.compute_type
            or new_config.stt.api_base_url != old.stt.api_base_url
            or new_config.stt.api_key != old.stt.api_key
            or new_config.stt.api_model != old.stt.api_model
        )
        if stt_changed:
            logger.info("STT settings changed — reinitialising engine...")
            old_stt = self._stt
            self._stt = self._init_stt()
            # Clean up old STT engine (all engines have close() method)
            if hasattr(old_stt, "close"):
                old_stt.close()

        # --- LLM client ---
        llm_changed = (
            new_config.llm.base_url != old.llm.base_url
            or new_config.llm.api_key != old.llm.api_key
            or new_config.llm.model != old.llm.model
            or new_config.llm.temperature != old.llm.temperature
            or new_config.llm.max_tokens != old.llm.max_tokens
            or new_config.llm.prompts.polish != old.llm.prompts.polish
            or new_config.llm.prompts.insert != old.llm.prompts.insert
        )
        if llm_changed:
            logger.info("LLM settings changed — reinitialising client")
            if self._llm is not None:
                self._llm.close()
            self._llm = self._init_llm_client()

        # --- Overlay capsule position mode ---
        if new_config.overlay.capsule_position_mode != old.overlay.capsule_position_mode:
            logger.info(
                "Capsule position mode changed (%r→%r)",
                old.overlay.capsule_position_mode,
                new_config.overlay.capsule_position_mode,
            )
            self._overlay.set_capsule_position_mode(new_config.overlay.capsule_position_mode)

        # --- Personas ---
        self._personas = load_personas()
        logger.info("Reloaded %d persona(s)", len(self._personas))

        logger.info("Settings update complete")

    def _on_personas_changed(self) -> None:
        """Handle persona changes pushed from the persona manager dialog."""
        self._personas = load_personas()
        logger.info("Reloaded %d persona(s) from persona manager", len(self._personas))

    def _on_rerun_wizard(self) -> None:
        """Handle rerun wizard request from settings dialog."""
        logger.info("Rerunning setup wizard...")
        try:
            # Import wizard module
            from untype.wizard import run_setup_wizard
            from untype.config import load_config

            config = load_config()

            def on_wizard_complete(updated_config: AppConfig) -> None:
                logger.info("Setup wizard completed, configuration updated")
                # Reload configuration
                self._config = updated_config
                self._prev_config = copy.deepcopy(updated_config)
                # Reinitialize components that depend on config
                self._hotkey.stop()
                self._hotkey = HotkeyListener(
                    self._config.hotkey.trigger, self._config.hotkey.mode, self._on_hotkey_press
                )
                self._hotkey.start()
                logger.info("Hotkey listener restarted")

            # Run wizard (blocking call)
            run_setup_wizard(config, on_wizard_complete)
        except Exception as e:
            logger.exception("Failed to rerun wizard: %s", e)

    def _on_capsule_position_changed(self, x: int, y: int) -> None:
        """Handle capsule position change from drag (fixed mode)."""
        self._config.overlay.capsule_fixed_x = x
        self._config.overlay.capsule_fixed_y = y
        logger.debug("Queueing capsule position save: %d, %d", x, y)
        self._schedule_config_save(
            delay=_CONFIG_SAVE_DEBOUNCE_S,
            reason="capsule position",
        )

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        """Handle the Quit action from the tray menu."""
        logger.info("Shutting down...")
        self._flush_pending_config_save()
        self._hotkey.stop()
        self._digit_interceptor.stop()
        self._overlay.stop()
        if self._llm is not None:
            self._llm.close()
        if isinstance(self._stt, (STTApiEngine, STTRealtimeApiEngine)):
            self._stt.close()
        self._tray.stop()
        logger.info("Goodbye.")

    # ------------------------------------------------------------------
    # Module initialisation helpers
    # ------------------------------------------------------------------

    def _init_recorder(self) -> AudioRecorder:
        """Create an AudioRecorder from the current config."""
        device = self._config.audio.device or None  # "" -> None (system default)
        return AudioRecorder(
            sample_rate=self._config.audio.sample_rate,
            device=device,
            on_volume=self._on_audio_volume,
            on_audio_chunk=self._on_audio_chunk,
        )

    def _start_timeout_monitor(self) -> None:
        """Start a background thread to monitor recording duration and enforce timeout."""
        # Stop any existing timer
        self._stop_timeout_timer.set()

        # Reset the stop event for the new timer
        self._stop_timeout_timer.clear()

        # Start a new timer thread
        self._timeout_timer = threading.Thread(
            target=self._timeout_monitor_loop,
            name="untype-timeout-monitor",
            daemon=True,
        )
        self._timeout_timer.start()

    def _stop_timeout_monitor(self) -> None:
        """Stop the timeout monitor thread."""
        self._stop_timeout_timer.set()

    def _timeout_monitor_loop(self) -> None:
        """Background loop that monitors recording duration and enforces timeout.

        Runs every second while recording is active:
        - Updates duration display on capsule
        - Shows warning when approaching timeout
        - Auto-stops recording when max duration is reached
        """
        from untype.audio import AudioRecorder

        MAX_SECONDS = AudioRecorder.MAX_RECORDING_SECONDS  # 5 minutes
        WARNING_SECONDS = 30  # Show warning 30 seconds before timeout

        while not self._stop_timeout_timer.is_set():
            if self._recorder.is_recording:
                duration = self._recorder.get_duration()
                remaining = MAX_SECONDS - duration

                # Update duration display
                warning = remaining <= WARNING_SECONDS
                self._overlay.update_duration(duration, warning=warning)

                # Log warning when approaching timeout
                if warning and int(duration) % 10 == 0:
                    logger.warning(
                        "Recording duration: %.0fs (timeout in %d seconds)",
                        duration,
                        int(remaining),
                    )

                # Auto-stop when timeout is reached
                if duration >= MAX_SECONDS:
                    logger.warning(
                        "Recording timeout reached (%d seconds), auto-stopping",
                        MAX_SECONDS,
                    )
                    # Trigger cancel which will stop recording
                    self._on_cancel()
                    break

            # Check every second
            self._stop_timeout_timer.wait(timeout=1.0)

    def _on_audio_volume(self, level: float) -> None:
        """Handle audio volume update from the recorder (called from audio thread)."""
        self._overlay.update_volume(level)

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Handle audio chunk during recording for realtime STT (called from audio thread)."""
        if self._config.stt.backend == "realtime_api":
            if isinstance(self._stt, STTRealtimeApiEngine):
                self._stt.on_audio_chunk(chunk)

    def _on_realtime_text_update(self, text: str) -> None:
        """Handle realtime text update from streaming STT (called from STT thread).

        Updates the live transcription preview on the overlay during recording.
        """
        logger.debug("Realtime text update: %s", text)
        self._overlay.update_realtime_preview(text)

    def _handle_stt_config_check(self) -> None:
        """Handle STT configuration self-check at startup."""
        cfg = self._config.stt

        # Local model: check if exists, offer to download if not
        if cfg.backend == "local":
            if not self._check_local_model_exists(cfg.model_size):
                self._handle_model_download(cfg.model_size)
            return

        # Online API: check if configured, offer to open settings if not
        if cfg.backend == "api":
            missing = []
            if not cfg.api_base_url:
                missing.append("API 地址")
            if not cfg.api_key:
                missing.append("API 密钥")
            if missing:
                self._handle_missing_api_config(missing)
            return

        # Realtime API: check if configured, offer to open settings if not
        if cfg.backend == "realtime_api":
            api_key = cfg.realtime_api_key or cfg.api_key
            if not api_key:
                self._handle_missing_api_config(["阿里云 API 密钥"])
            return

    def _handle_model_download(self, model_size: str) -> None:
        """Handle the case when local model needs to be downloaded."""
        import tkinter.messagebox as messagebox
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        # Check network first
        has_network = self._check_network_connectivity()

        if not has_network:
            messagebox.showerror(
                "无法连接网络",
                f"本地模型 '{model_size}' 不存在，需要从网络下载。\n\n"
                "但当前无法连接到 HuggingFace，请检查网络环境后重试。"
            )
            root.destroy()
            return

        # Network OK, ask user to confirm download
        result = messagebox.askyesno(
            "下载本地模型",
            f"本地模型 '{model_size}' 不存在，需要从网络下载（约 500MB）。\n\n"
            "请确保网络已连接，点击「是」开始下载。\n\n"
            "下载可能需要几分钟时间，请耐心等待。"
        )
        root.destroy()

        if result:
            self._download_whisper_model(model_size)

    def _download_whisper_model(self, model_size: str) -> None:
        """Download the Whisper model and show progress."""
        import tkinter as tk
        from tkinter import ttk

        # Create progress window
        progress_root = tk.Tk()
        progress_root.title("下载模型")
        progress_root.geometry("400x150")
        progress_root.resizable(False, False)

        # Center the window
        progress_root.update_idletasks()
        w = progress_root.winfo_width()
        h = progress_root.winfo_height()
        x = (progress_root.winfo_screenwidth() - w) // 2
        y = (progress_root.winfo_screenheight() - h) // 2
        progress_root.geometry(f"+{x}+{y}")

        # Label
        label = tk.Label(
            progress_root,
            text=f"正在下载 Whisper 模型 ({model_size})...\n请稍候",
            font=("Microsoft YaHei UI", 10)
        )
        label.pack(pady=20)

        # Progress bar
        progress_bar = ttk.Progressbar(
            progress_root,
            mode="indeterminate",
            length=300
        )
        progress_bar.pack(pady=10)
        progress_bar.start()

        # Status label
        status_label = tk.Label(
            progress_root,
            text="正在连接 HuggingFace...",
            font=("Microsoft YaHei UI", 9),
            fg="#666666"
        )
        status_label.pack(pady=5)

        # Download in a separate thread
        def download_thread():
            try:
                from faster_whisper import WhisperModel

                # Update status
                progress_root.after(0, lambda: status_label.configure(text="正在下载模型文件..."))

                # This will download the model
                model = WhisperModel(model_size, device="cpu", compute_type="int8")

                # Success
                progress_root.after(0, lambda: status_label.configure(text="下载完成！"))
                progress_root.after(0, lambda: progress_bar.stop())
                progress_root.after(2000, lambda: progress_root.destroy())

                logger.info("Whisper model %s downloaded successfully", model_size)

            except Exception as e:
                # Error
                progress_root.after(0, lambda: progress_bar.stop())
                progress_root.after(0, lambda: status_label.configure(text=f"下载失败: {e}"))
                progress_root.after(0, lambda: status_label.configure(fg="#ff6666"))
                progress_root.after(
                    0,
                    lambda: messagebox.showerror(
                        "下载失败",
                        f"模型下载失败：{e}\n\n请检查网络连接后重试。"
                    )
                )
                progress_root.after(0, lambda: progress_root.destroy())
                logger.exception("Failed to download Whisper model")

        import threading
        thread = threading.Thread(target=download_thread, daemon=True)
        thread.start()

        progress_root.mainloop()

    def _handle_missing_api_config(self, missing_items: list[str]) -> None:
        """Handle the case when API configuration is missing."""
        import tkinter.messagebox as messagebox
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        missing_text = "、".join(missing_items)
        result = messagebox.askyesno(
            "配置未完成",
            f"当前模式需要配置 {missing_text}。\n\n点击「是」打开设置进行配置。"
        )
        root.destroy()

        if result:
            # Open settings dialog
            self._tray.show_settings()

    def _check_local_model_exists(self, model_size: str) -> bool:
        """Check if the local Whisper model already exists."""
        try:
            from faster_whisper import WhisperModel
            # Try to load the model without downloading
            # This will raise an error if the model doesn't exist locally
            import os
            # Check common cache locations
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if not os.path.exists(cache_dir):
                return False

            # Look for model folder in cache
            model_name = f"Systran/faster-whisper-{model_size}"
            for item in os.listdir(cache_dir):
                if model_name.replace("-", "_").replace("/", "--") in item or model_name in item:
                    return True
            return False
        except Exception:
            return False

    def _check_network_connectivity(self, timeout: float = 3.0) -> bool:
        """Check if network connectivity to HuggingFace is available."""
        try:
            import urllib.request
            import socket
            socket.setdefaulttimeout(timeout)
            urllib.request.urlopen("https://huggingface.co", timeout=timeout)
            return True
        except Exception:
            return False

    def _validate_api_endpoint(self, base_url: str, api_key: str) -> bool:
        """Validate if the API endpoint is accessible with the given key.

        This is a simple check - we just verify the endpoint is reachable.
        """
        try:
            import httpx
            import urllib.parse

            # Parse and validate URL
            parsed = urllib.parse.urlparse(base_url)
            if not parsed.scheme or not parsed.netloc:
                return False

            # Try a simple health check or OPTIONS request
            with httpx.Client(timeout=5.0) as client:
                response = client.options(
                    base_url,
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                # Any response (including 401/403) means the server is reachable
                return True
        except Exception as e:
            logger.debug("API endpoint validation failed: %s", e)
            return False

    def _validate_dashscope_key(self, api_key: str) -> bool:
        """Validate if the DashScope API key is format-valid and service is reachable."""
        try:
            import httpx

            # DashScope keys start with "sk-"
            if not api_key.startswith("sk-"):
                return False

            # Try to reach the DashScope API
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    "https://dashscope.aliyuncs.com/api/v1/services",
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                # Any response (including 401/403/404) means the service is reachable
                return True
        except Exception as e:
            logger.debug("DashScope key validation failed: %s", e)
            return False

    def _init_stt(self) -> STTEngine | STTApiEngine | STTRealtimeApiEngine:
        """Create an STT engine from the current config."""
        cfg = self._config.stt

        # Fallback to API if local STT is selected but not available
        if cfg.backend == "local" and not HAS_LOCAL_STT:
            logger.warning(
                "Local STT backend selected but not available in this build. "
                "Falling back to API backend."
            )
            cfg.backend = "api"

        if cfg.backend == "api":
            logger.info("Using API STT backend (%s)", cfg.api_model)
            return STTApiEngine(
                base_url=cfg.api_base_url,
                api_key=cfg.api_key,
                model=cfg.api_model,
                language=cfg.language,
                sample_rate=self._config.audio.sample_rate,
            )

        if cfg.backend == "realtime_api":
            api_key = cfg.realtime_api_key or cfg.api_key
            logger.info("Using Realtime API STT backend (%s)", cfg.realtime_api_model)
            return STTRealtimeApiEngine(
                api_key=api_key,
                model=cfg.realtime_api_model,
                language=cfg.language,
                format=cfg.realtime_api_format,
                sample_rate=cfg.realtime_api_sample_rate,
                on_text_update=self._on_realtime_text_update,
            )

        logger.info("Using local STT backend (model=%s)", cfg.model_size)
        return STTEngine(
            model_size=cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
            language=cfg.language,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            vad_threshold=cfg.vad_threshold,
        )

    def _init_llm_client(self) -> LLMClient | None:
        """Create an LLM client from config.

        Returns ``None`` if the LLM is not fully configured (missing
        base_url, api_key, or model), in which case pipeline steps that
        require the LLM will fall back to raw transcription output.
        """
        cfg = self._config.llm

        if not cfg.base_url or not cfg.api_key or not cfg.model:
            logger.warning(
                "LLM not fully configured (base_url=%r, api_key=%s, model=%r) "
                "— LLM processing will be skipped",
                cfg.base_url,
                "***" if cfg.api_key else "(empty)",
                cfg.model,
            )
            return None

        return LLMClient(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            prompts={
                "polish": cfg.prompts.polish,
                "insert": cfg.prompts.insert,
            },
        )


# ======================================================================
# Entry point
# ======================================================================


def main() -> None:
    """Entry point for the UnType application."""
    # Setup logging
    _setup_logging()

    # Check if first run and show setup wizard
    if is_first_run():
        logger.info("First run detected, launching setup wizard...")
        _run_first_run_wizard()

    app = UnTypeApp()
    app.run()


def is_first_run() -> bool:
    """Check if this is the first run of UnType."""
    from untype.wizard import is_first_run as _is_first_run
    return _is_first_run()


def _run_first_run_wizard() -> None:
    """Run the first-run setup wizard."""
    from untype.wizard import run_setup_wizard
    from untype.config import load_config

    config = load_config()

    def on_wizard_complete(updated_config: AppConfig) -> None:
        logger.info("Setup wizard completed, configuration updated")
        _show_tray_notification()

    run_setup_wizard(config, on_wizard_complete)


def _show_tray_notification() -> None:
    """Show a bubble notification near the system tray after wizard completion."""
    import threading

    def _show_notification() -> None:
        try:
            import tkinter as tk
            import time

            # Wait for wizard to fully close and main app to start
            time.sleep(1.5)

            # Create notification window
            notif = tk.Tk()
            notif.configure(bg="#2d2d2d")
            notif.withdraw()
            notif.attributes("-topmost", True)
            notif.attributes("-toolwindow", True)
            notif.overrideredirect(True)

            # Dark theme colors
            bg_color = "#2d2d2d"
            fg_color = "#e0e0e0"
            accent_color = "#4CAF50"

            # Main frame
            main = tk.Frame(notif, bg=bg_color, relief="solid", borderwidth=1)
            main.pack(padx=0, pady=0)

            # Content
            content = tk.Frame(main, bg=bg_color)
            content.pack(fill="both", expand=True, padx=20, pady=15)

            # Icon and title
            top = tk.Frame(content, bg=bg_color)
            top.pack(fill="x", pady=(0, 10))

            tk.Label(
                top,
                text="✅ 配置完成！",
                font=("Microsoft YaHei UI", 11, "bold"),
                bg=bg_color,
                fg=fg_color,
            ).pack()

            tk.Label(
                content,
                text="我在右下角状态栏 👇",
                font=("Microsoft YaHei UI", 9),
                bg=bg_color,
                fg="#b0bec5",
            ).pack(pady=(0, 5))

            tk.Label(
                content,
                text="按 F6 开始使用",
                font=("Microsoft YaHei UI", 9),
                bg=bg_color,
                fg=accent_color,
            ).pack()

            # Position at bottom right
            notif.update_idletasks()
            screen_w = notif.winfo_screenwidth()
            screen_h = notif.winfo_screenheight()

            notif_width = 220
            notif_height = 110

            x = screen_w - notif_width - 15
            y = screen_h - notif_height - 50

            notif.geometry(f"{notif_width}x{notif_height}+{x}+{y}")
            notif.deiconify()

            # Auto-dismiss with cleanup
            def close_notif():
                try:
                    notif.destroy()
                except Exception:
                    pass

            notif.after(5000, close_notif)
            notif.mainloop()

        except Exception as e:
            logger.warning("Failed to show tray notification: %s", e)

    # Run in daemon thread so it doesn't block the main app
    thread = threading.Thread(target=_show_notification, daemon=True)
    thread.start()


def _setup_logging() -> None:
    """Configure logging with console and file handlers.

    Log files are stored in ~/.untype/logs/ with rotation to keep size manageable.
    """
    # Create logs directory
    log_dir = _get_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    # Common log format
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (with colored output for Windows)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    # Keep 3 backup files, max 500KB each = ~2MB total
    log_file = os.path.join(log_dir, "untype.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=500 * 1024,  # 500KB per file
        backupCount=3,  # Keep 3 backup files
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # File gets more detailed logs
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logging.info("UnType starting...")
    logging.info("Log file: %s", log_file)


def _get_log_dir() -> str:
    """Get the log directory path.

    Returns:
        Path to the logs directory (e.g., C:\\Users\\xxx\\.untype\\logs).
    """
    home = os.path.expanduser("~")
    log_dir = os.path.join(home, ".untype", "logs")
    return log_dir


def get_log_file_path() -> str:
    """Get the full path to the current log file.

    This can be used to show the user where logs are stored.

    Returns:
        Full path to the untype.log file.
    """
    return os.path.join(_get_log_dir(), "untype.log")


if __name__ == "__main__":
    main()
