"""Main entry point — orchestrates all UnType modules together."""

from __future__ import annotations

import copy
import logging
import logging.handlers
import os
import threading
import time

import numpy as np
import pyperclip

from untype.audio import AudioRecorder, normalize_audio
from untype.clipboard import grab_selected_text, inject_text, release_all_modifiers
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

logger = logging.getLogger(__name__)


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
            logger.info(
                "Loaded %d persona(s): %s",
                len(self._personas),
                ", ".join(p.name for p in self._personas),
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

        # -- Initialise subsystems -------------------------------------------
        logger.info("Initialising audio recorder...")
        self._recorder = self._init_recorder()

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

        logger.info("Initialising system tray...")
        self._tray = TrayApp(
            config=self._config,
            on_settings_changed=self._on_settings_changed,
            on_quit=self._on_quit,
            on_personas_changed=self._on_personas_changed,
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
            self._held_result = None
            self._held_clipboard = None
            self._overlay.hide_hold_bubble()

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

        # Deactivate digit interceptor.
        self._digit_interceptor.set_active(False)

        # Stop HWND watcher.
        self._hwnd_watch_active = False

        # Hide all overlays.
        self._overlay.hide()
        self._overlay.hide_recording_personas()
        self._overlay.hide_hold_bubble()

        # Force-unblock staging if it's waiting.
        self._overlay._staging_result_action = "cancel"
        self._overlay._staging_result_text = ""
        self._overlay._staging_event.set()

        # Update tray status.
        self._tray.update_status("Ready")

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

            # CRITICAL: Ensure recorder is stopped even if _process_pipeline won't run
            if self._recorder.is_recording:
                try:
                    self._recorder.abort()
                except Exception:
                    pass

            # Also stop realtime STT session if active
            try:
                if isinstance(self._stt, STTRealtimeApiEngine):
                    self._stt.stop_session()
            except Exception:
                pass
        finally:
            # Always release the lock
            self._hwnd_watch_active = False
            self._digit_interceptor.set_active(False)
            self._cancel_requested.clear()
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

            # Start recording FIRST so the user's speech is captured from
            # the moment they press the hotkey, not after the clipboard probe.
            logger.info("Starting audio recording...")
            self._recorder.start()
            self._tray.update_status("Recording...")
            self._overlay.show(self._caret_x, self._caret_y, "Recording...")

            # Start the timeout monitor thread
            self._start_timeout_monitor()

            # Show recording persona bar (if personas configured).
            self._preselected_persona = None
            if self._personas:
                persona_tuples = [(p.id, p.icon, p.name) for p in self._personas]
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
                    for idx, p in enumerate(self._personas):
                        if p.id == last_id:
                            self._preselected_persona = p
                            self._overlay.select_recording_persona(idx)
                            logger.info("Auto-selected remembered persona: %s", p.name)
                            break

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

            # Start realtime recognition session AFTER _recording_started is set,
            # in a separate thread to avoid blocking. This allows the pipeline to
            # check for cancellation before the WebSocket connection is established.
            if self._config.stt.backend == "realtime_api":

                def _start_realtime_session():
                    if not self._cancel_requested.is_set():
                        try:
                            if isinstance(self._stt, STTRealtimeApiEngine):
                                self._stt.start_recording_session()
                                self._overlay.show_realtime_preview(self._caret_x, self._caret_y)
                        except Exception as e:
                            logger.warning("Failed to start realtime STT session: %s", e)

                threading.Thread(
                    target=_start_realtime_session,
                    daemon=True,
                    name="untype-start-realtime",
                ).start()

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
                # CRITICAL: Must abort the recorder immediately when cancelled
                # Use ThreadPoolExecutor with timeout to avoid blocking
                import concurrent.futures

                def _cleanup_resources():
                    if self._recorder.is_recording:
                        logger.debug("Cleanup: aborting recorder")
                        self._recorder.abort()
                    if isinstance(self._stt, STTRealtimeApiEngine):
                        logger.debug("Cleanup: stopping STT session")
                        self._stt.stop_session()
                    logger.debug("Cleanup: complete")

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(_cleanup_resources)
                    future.result(timeout=2.0)  # Max 2 seconds for cleanup
                except concurrent.futures.TimeoutError:
                    logger.warning("Cleanup timed out - proceeding anyway")
                except Exception as e:
                    logger.warning("Cleanup error: %s", e)
                finally:
                    executor.shutdown(wait=False)
                return

            if not self._recorder.is_recording:
                logger.warning("Recording not active — aborting pipeline")
                self._digit_interceptor.set_active(False)
                self._overlay.hide_recording_personas()
                self._tray.update_status("Ready")
                self._overlay.hide()
                return

            # Checkpoint: cancel requested before stopping recording?
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled before recording stop - initiating cleanup")
                # CRITICAL: Must abort the recorder immediately when cancelled
                # Use ThreadPoolExecutor with timeout to avoid blocking
                import concurrent.futures

                def _cleanup_resources():
                    if self._recorder.is_recording:
                        logger.debug("Cleanup: aborting recorder")
                        self._recorder.abort()
                    if isinstance(self._stt, STTRealtimeApiEngine):
                        logger.debug("Cleanup: stopping STT session")
                        self._stt.stop_session()

                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(_cleanup_resources)
                    future.result(timeout=2.0)  # Max 2 seconds for cleanup
                except concurrent.futures.TimeoutError:
                    logger.warning("Cleanup timed out - proceeding anyway")
                except Exception as e:
                    logger.warning("Cleanup error: %s", e)
                finally:
                    executor.shutdown(wait=False)

                self._digit_interceptor.set_active(False)
                self._overlay.hide_recording_personas()
                self._overlay.hide_realtime_preview()
                self._tray.update_status("Ready")
                self._overlay.hide()
                return

            # 1. Stop recording and retrieve audio.
            logger.info("Stopping audio recording...")
            self._stop_timeout_monitor()  # Stop timeout monitor before stopping recorder
            audio = self._recorder.stop()

            # Checkpoint: cancel requested during recording stop?
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled after recording stop")
                return

            if audio.size == 0:
                logger.warning("Empty audio buffer — aborting pipeline")
                self._digit_interceptor.set_active(False)
                self._overlay.hide_recording_personas()
                self._overlay.hide_realtime_preview()
                self._tray.update_status("Ready")
                self._overlay.hide()
                return

            # 2. Transcribe (or get realtime result).
            # IMPORTANT: Check cancel before STT processing
            if self._cancel_requested.is_set():
                logger.info("Pipeline cancelled before STT processing")
                # Still need to cleanup
                try:
                    if isinstance(self._stt, STTRealtimeApiEngine):
                        self._stt.stop_session()
                except Exception:
                    pass
                self._digit_interceptor.set_active(False)
                self._overlay.hide_recording_personas()
                self._overlay.hide_realtime_preview()
                self._tray.update_status("Ready")
                self._overlay.hide()
                return

            if self._config.stt.backend == "realtime_api":
                # For realtime API, use ThreadPoolExecutor to prevent blocking
                logger.info("Stopping realtime recognition session...")
                if isinstance(self._stt, STTRealtimeApiEngine):
                    import concurrent.futures

                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = executor.submit(self._stt.stop_session)
                        text = future.result(timeout=3.0)  # 3 second timeout
                    except concurrent.futures.TimeoutError:
                        logger.warning("STT stop_session timed out - forcing cleanup")
                        text = self._stt.get_result()
                    finally:
                        executor.shutdown(wait=False)
                else:
                    text = ""
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
                return

            if not text:
                logger.warning("Empty transcription — aborting pipeline")
                self._digit_interceptor.set_active(False)
                self._overlay.hide_recording_personas()
                self._overlay.hide_realtime_preview()
                self._tray.update_status("Ready")
                self._overlay.hide()
                return

            logger.info("Transcription: %s", text)

            # 4. STT complete — deactivate digit interceptor and hide persona bar.
            self._digit_interceptor.set_active(False)
            self._overlay.hide_recording_personas()

            # 5. Branch: personas configured → skip staging; otherwise → staging.
            if self._personas:
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

        if at_corner:
            # Capsule is already parked at the corner from a prior HWND
            # mismatch — just update its status text in place.
            self._overlay.update_status("Processing...")
        else:
            self._overlay.show(self._caret_x, self._caret_y, "Processing...")
        self._tray.update_status("Processing...")

        # Restart HWND monitoring during the LLM call.
        self._start_hwnd_watcher()

        # Check if cancel was already requested before starting LLM
        if self._cancel_requested.is_set():
            logger.info("Pipeline cancelled before LLM call (fast-lane)")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return

        try:
            result = self._run_llm(text, persona=persona)
        except KeyboardInterrupt:
            # LLM request was cancelled by user
            logger.info("Pipeline cancelled during LLM (fast-lane)")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return
        except Exception as e:
            logger.exception(f"Pipeline error in _process_with_personas: {e}")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return

        # IMPORTANT: Check cancel again immediately after LLM returns
        # User may have cancelled while LLM was processing (HTTP may have completed)
        if self._cancel_requested.is_set():
            logger.info("Pipeline cancelled after LLM returned (fast-lane)")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return

        # Stop watcher and verify window before injection.
        self._hwnd_watch_active = False
        if not self._verify_window_safety():
            logger.warning(
                "Window changed during LLM processing — holding result",
            )
            self._save_interaction_state(text, result, persona=persona, show_ghost=False)
            self._held_result = result
            self._held_clipboard = self._original_clipboard
            self._overlay.fly_to_hold_bubble(result)
            self._tray.update_status("Ready")
            return

        logger.info("Injecting refined text (%d chars)", len(result))
        inject_text(result, self._original_clipboard)
        self._save_interaction_state(text, result, persona=persona)
        self._overlay.hide()
        self._tray.update_status("Ready")
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
            inject_text(edited_text, self._original_clipboard)
            self._save_interaction_state(text, edited_text)
            return

        # action == "refine" — send through LLM.
        self._overlay.show(self._caret_x, self._caret_y, "Processing...")
        self._tray.update_status("Processing...")

        # Restart HWND monitoring during the LLM call.
        self._start_hwnd_watcher()

        try:
            result = self._run_llm(edited_text)
        except KeyboardInterrupt:
            # LLM request was cancelled by user
            logger.info("Pipeline cancelled during LLM (staging)")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return
        except Exception as e:
            logger.exception(f"Pipeline error in _process_with_staging LLM: {e}")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return

        # IMPORTANT: Check cancel again immediately after LLM returns
        if self._cancel_requested.is_set():
            logger.info("Pipeline cancelled after LLM returned (staging)")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
            return

        # Stop watcher and verify window before injection.
        self._hwnd_watch_active = False
        if not self._verify_window_safety():
            logger.warning(
                "Window changed during LLM processing — holding result",
            )
            self._save_interaction_state(text, result, show_ghost=False)
            self._held_result = result
            self._held_clipboard = self._original_clipboard
            self._overlay.fly_to_hold_bubble(result)
            self._tray.update_status("Ready")
            return

        logger.info("Injecting refined text (%d chars)", len(result))
        inject_text(result, self._original_clipboard)
        self._save_interaction_state(text, result)
        self._overlay.hide()
        self._tray.update_status("Ready")
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

    # ------------------------------------------------------------------
    # Hold callbacks (Phase 2 — called from overlay thread)
    # ------------------------------------------------------------------

    def _on_hold_inject(self) -> None:
        """Left-click on hold bubble — inject into the current foreground window."""
        result = self._held_result
        clipboard = self._held_clipboard
        self._held_result = None
        self._held_clipboard = None

        if result is None:
            return

        logger.info("Hold-inject: injecting %d chars into current window", len(result))
        inject_text(result, clipboard)

        # Show ghost menu if we have saved interaction state.
        if self._last_raw_text is not None:
            self._last_result = result
            caret = get_caret_screen_position()
            self._last_caret_x = caret.x
            self._last_caret_y = caret.y
            self._last_target_window = get_foreground_window()
            self._overlay.show_ghost_menu(caret.x, caret.y)

    def _on_hold_copy(self) -> None:
        """Right-click on hold bubble — copy held text to clipboard."""
        result = self._held_result
        self._held_result = None
        self._held_clipboard = None

        if result is None:
            return

        logger.info("Hold-copy: copying %d chars to clipboard", len(result))
        try:
            pyperclip.copy(result)
        except Exception:
            logger.exception("Failed to copy held result to clipboard")

    def _on_hold_ghost(self) -> None:
        """Middle-click on hold bubble — show ghost menu for revert/regenerate.

        Does NOT inject text.  The held result is discarded (the ghost
        menu actions will use ``_last_*`` state instead).
        """
        self._held_result = None
        self._held_clipboard = None

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
        self._last_target_window = self._target_window

        caret = get_caret_screen_position()
        self._last_caret_x = caret.x
        self._last_caret_y = caret.y

        if show_ghost:
            self._overlay.show_ghost_menu(caret.x, caret.y)

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

        if not self._pipeline_lock.acquire(blocking=False):
            logger.warning("Ghost revert: pipeline busy — ignoring")
            return

        try:
            # Undo the paste if the target window is still in foreground.
            target = self._last_target_window
            if target is not None and verify_foreground_window(target):
                logger.info("Ghost revert: undoing paste via Ctrl+Z")
                self._simulate_undo()
            else:
                logger.info("Ghost revert: target window not in foreground, skipping undo")

            # Restore interaction context for the staging area.
            self._mode = self._last_mode or "insert"
            self._selected_text = self._last_selected_text
            self._original_clipboard = self._last_original_clipboard
            self._target_window = target
            self._caret_x = self._last_caret_x
            self._caret_y = self._last_caret_y
            self._window_mismatch = False

            # Build persona list for staging (if available).
            personas_arg = None
            if self._personas:
                personas_arg = [(p.id, p.icon, p.name) for p in self._personas]

            # Clear last state before re-entering staging.
            self._last_raw_text = None
            self._last_result = None

            # Show staging area with the raw text.
            self._overlay.show_staging(
                raw_text,
                self._caret_x,
                self._caret_y,
                personas=personas_arg,
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
                inject_text(edited_text, self._original_clipboard)
                self._save_interaction_state(raw_text, edited_text)
                return

            # action == "refine"
            self._overlay.show(self._caret_x, self._caret_y, "Processing...")
            self._tray.update_status("Processing...")

            # Start HWND monitoring during LLM call.
            self._start_hwnd_watcher()

            try:
                result = self._run_llm(edited_text, persona=persona)
            except KeyboardInterrupt:
                # LLM request was cancelled by user
                logger.info("Ghost revert: LLM cancelled by user")
                self._hwnd_watch_active = False
                self._overlay.hide()
                self._tray.update_status("Ready")
                return

            # Stop watcher and verify window before injection.
            self._hwnd_watch_active = False
            if not self._verify_window_safety():
                logger.warning(
                    "Ghost revert: window changed during LLM — holding result",
                )
                self._save_interaction_state(
                    raw_text,
                    result,
                    persona=persona,
                    show_ghost=False,
                )
                self._held_result = result
                self._held_clipboard = self._original_clipboard
                self._overlay.fly_to_hold_bubble(result)
                self._tray.update_status("Ready")
                return

            logger.info("Ghost revert: injecting refined text (%d chars)", len(result))
            inject_text(result, self._original_clipboard)
            self._save_interaction_state(raw_text, result, persona=persona)
            self._overlay.hide()
            self._tray.update_status("Ready")

        except Exception:
            logger.exception("Ghost revert error")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
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

        if not self._pipeline_lock.acquire(blocking=False):
            logger.warning("Ghost regenerate: pipeline busy — ignoring")
            return

        try:
            # Undo the paste if the target window is still in foreground.
            target = self._last_target_window
            if target is not None and verify_foreground_window(target):
                logger.info("Ghost regenerate: undoing paste via Ctrl+Z")
                self._simulate_undo()
            else:
                logger.info("Ghost regenerate: target window not in foreground, skipping undo")

            # Restore interaction context.
            self._mode = self._last_mode or "insert"
            self._selected_text = self._last_selected_text
            self._original_clipboard = self._last_original_clipboard
            self._target_window = target
            self._caret_x = self._last_caret_x
            self._caret_y = self._last_caret_y
            self._window_mismatch = False

            # Show capsule with processing status.
            self._overlay.show(self._caret_x, self._caret_y, "Processing...")
            self._tray.update_status("Processing...")

            # Start HWND monitoring during LLM call.
            self._start_hwnd_watcher()

            try:
                result = self._run_llm(raw_text, persona=persona)
            except KeyboardInterrupt:
                # LLM request was cancelled by user
                logger.info("Ghost regenerate: LLM cancelled by user")
                self._hwnd_watch_active = False
                self._overlay.hide()
                self._tray.update_status("Ready")
                return

            # Stop watcher and verify window before injection.
            self._hwnd_watch_active = False
            if not self._verify_window_safety():
                logger.warning(
                    "Ghost regenerate: window changed during LLM — holding result",
                )
                self._save_interaction_state(
                    raw_text,
                    result,
                    persona=persona,
                    show_ghost=False,
                )
                self._held_result = result
                self._held_clipboard = self._original_clipboard
                self._overlay.fly_to_hold_bubble(result)
                self._tray.update_status("Ready")
                return

            logger.info("Ghost regenerate: injecting refined text (%d chars)", len(result))
            inject_text(result, self._original_clipboard)
            self._save_interaction_state(raw_text, result, persona=persona)
            self._overlay.hide()
            self._tray.update_status("Ready")

        except Exception:
            logger.exception("Ghost regenerate error")
            self._hwnd_watch_active = False
            self._overlay.hide()
            self._tray.update_status("Ready")
        finally:
            self._hwnd_watch_active = False
            self._pipeline_lock.release()

    def _on_ghost_use_raw(self) -> None:
        """Ghost menu 'Use Raw' — undo paste, inject raw STT text instead."""
        raw_text = self._last_raw_text
        if raw_text is None:
            logger.warning("Ghost use-raw: no interaction state saved")
            return

        if not self._pipeline_lock.acquire(blocking=False):
            logger.warning("Ghost use-raw: pipeline busy — ignoring")
            return

        try:
            # Undo the paste if the target window is still in foreground.
            target = self._last_target_window
            if target is not None and verify_foreground_window(target):
                logger.info("Ghost use-raw: undoing paste via Ctrl+Z")
                self._simulate_undo()
            else:
                logger.info("Ghost use-raw: target window not in foreground, skipping undo")

            # Restore context.
            self._original_clipboard = self._last_original_clipboard
            self._target_window = target

            logger.info("Ghost use-raw: injecting raw text (%d chars)", len(raw_text))
            inject_text(raw_text, self._original_clipboard)
            self._save_interaction_state(raw_text, raw_text)

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
        if idx < len(self._personas):
            if self._preselected_persona == self._personas[idx]:
                # Toggle off: pressing the same digit again deselects.
                self._preselected_persona = None
                self._overlay.select_recording_persona(-1)
                self._save_selected_persona(None)
            else:
                self._preselected_persona = self._personas[idx]
                self._overlay.select_recording_persona(idx)
                self._save_selected_persona(self._personas[idx].id)

    def _save_selected_persona(self, persona_id: str | None) -> None:
        """Save the selected persona ID to config."""
        from untype.config import save_config

        # Save "default" as empty string (the actual default)
        if persona_id == "default":
            persona_id = None

        self._config.last_selected_persona = persona_id or "default"
        # Spawn a thread to save config without blocking
        threading.Thread(
            target=save_config,
            args=(self._config,),
            name="untype-save-persona",
            daemon=True,
        ).start()

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
        save_config(new_config)

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

    def _on_capsule_position_changed(self, x: int, y: int) -> None:
        """Handle capsule position change from drag (fixed mode)."""
        self._config.overlay.capsule_fixed_x = x
        self._config.overlay.capsule_fixed_y = y
        # Save to config file
        try:
            save_config(self._config)
            logger.debug("Saved capsule position: %d, %d", x, y)
        except Exception:
            logger.exception("Failed to save capsule position")

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        """Handle the Quit action from the tray menu."""
        logger.info("Shutting down...")
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

    def _init_stt(self) -> STTEngine | STTApiEngine | STTRealtimeApiEngine:
        """Create an STT engine from the current config."""
        cfg = self._config.stt

        if cfg.backend == "api" and cfg.api_base_url and cfg.api_key:
            logger.info("Using API STT backend (%s)", cfg.api_model)
            return STTApiEngine(
                base_url=cfg.api_base_url,
                api_key=cfg.api_key,
                model=cfg.api_model,
                language=cfg.language,
                sample_rate=self._config.audio.sample_rate,
            )

        if cfg.backend == "realtime_api":
            # Use realtime_api_key if set, otherwise fall back to api_key
            api_key = cfg.realtime_api_key or cfg.api_key
            if api_key:
                logger.info("Using Realtime API STT backend (%s)", cfg.realtime_api_model)
                return STTRealtimeApiEngine(
                    api_key=api_key,
                    model=cfg.realtime_api_model,
                    language=cfg.language,
                    format=cfg.realtime_api_format,
                    sample_rate=cfg.realtime_api_sample_rate,
                    on_text_update=self._on_realtime_text_update,
                )
            else:
                logger.warning(
                    "Realtime API selected but no API key configured, falling back to local"
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

    app = UnTypeApp()
    app.run()


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
