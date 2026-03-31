from __future__ import annotations

import threading
from types import SimpleNamespace

from untype.clipboard import InjectionResult
from untype.config import AppConfig, Persona
from untype.main import UnTypeApp


class DummyOverlay:
    def __init__(self) -> None:
        self.hidden = False
        self.held: list[str] = []
        self.ghost_menu_positions: list[tuple[int, int]] = []
        self.shown: list[tuple[int, int, str]] = []
        self.updated_statuses: list[str] = []
        self.recording_personas_hidden = False
        self.realtime_preview_hidden = False
        self.hold_bubble_hidden = False
        self.staging_cancelled = False

    def show(self, x: int, y: int, status: str) -> None:
        self.shown.append((x, y, status))

    def update_status(self, status: str) -> None:
        self.updated_statuses.append(status)

    def hide(self) -> None:
        self.hidden = True

    def fly_to_hold_bubble(self, text: str) -> None:
        self.held.append(text)

    def show_ghost_menu(self, x: int, y: int) -> None:
        self.ghost_menu_positions.append((x, y))

    def hide_recording_personas(self) -> None:
        self.recording_personas_hidden = True

    def hide_realtime_preview(self) -> None:
        self.realtime_preview_hidden = True

    def hide_hold_bubble(self) -> None:
        self.hold_bubble_hidden = True

    def cancel_staging(self) -> None:
        self.staging_cancelled = True


class DummyTray:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def update_status(self, status: str) -> None:
        self.statuses.append(status)


class DummyDigitInterceptor:
    def __init__(self) -> None:
        self.states: list[bool] = []

    def set_active(self, active: bool) -> None:
        self.states.append(active)


def _make_app() -> tuple[UnTypeApp, list[tuple[str, str, object | None, bool]]]:
    app = object.__new__(UnTypeApp)
    app._overlay = DummyOverlay()
    app._tray = DummyTray()
    app._digit_interceptor = DummyDigitInterceptor()
    app._original_clipboard = "original"
    app._held_result = None
    app._held_clipboard = None
    app._held_recent_result_id = None
    app._recent_results_lock = threading.Lock()
    app._recent_results = []
    app._mode = "insert"
    app._target_window = None
    app._window_mismatch = False
    app._caret_x = 0
    app._caret_y = 0
    app._personas = []
    saved: list[tuple[str, str, object | None, bool]] = []

    def fake_save(raw_text: str, result: str, persona=None, show_ghost: bool = True) -> None:
        saved.append((raw_text, result, persona, show_ghost))

    app._save_interaction_state = fake_save
    return app, saved


def test_deliver_result_saves_and_hides_on_success(monkeypatch) -> None:
    from untype import main as main_module

    app, saved = _make_app()
    app._verify_window_safety = lambda: True

    monkeypatch.setattr(
        main_module,
        "inject_text",
        lambda text, clipboard: InjectionResult(True, True),
    )

    delivered = app._deliver_result("raw", "clean")

    assert delivered is True
    assert saved == [("raw", "clean", None, True)]
    assert len(app._recent_results) == 1
    assert app._recent_results[0].status == "injected"
    assert app._overlay.hidden is True
    assert app._overlay.held == []
    assert app._tray.statuses == ["Ready"]


def test_deliver_result_holds_when_window_is_unsafe(monkeypatch) -> None:
    from untype import main as main_module

    app, saved = _make_app()
    app._verify_window_safety = lambda: False

    called = False

    def fake_inject(text: str, clipboard: str | None) -> InjectionResult:
        nonlocal called
        called = True
        return InjectionResult(True, True)

    monkeypatch.setattr(main_module, "inject_text", fake_inject)

    delivered = app._deliver_result("raw", "clean")

    assert delivered is False
    assert called is False
    assert saved == [("raw", "clean", None, False)]
    assert app._held_result == "clean"
    assert app._held_clipboard == "original"
    assert app._held_recent_result_id == app._recent_results[0].id
    assert app._recent_results[0].status == "held"
    assert app._overlay.held == ["clean"]
    assert app._tray.statuses == ["Ready"]


def test_take_held_result_clears_state_and_can_hide_bubble() -> None:
    app, _saved = _make_app()
    app._held_result = "held"
    app._held_clipboard = "clip"
    app._held_recent_result_id = "recent-1"

    result, clipboard, recent_result_id = app._take_held_result(hide_bubble=True)

    assert result == "held"
    assert clipboard == "clip"
    assert recent_result_id == "recent-1"
    assert app._held_result is None
    assert app._held_clipboard is None
    assert app._held_recent_result_id is None
    assert app._overlay.hold_bubble_hidden is True


def test_restore_held_result_reinstates_bubble_state() -> None:
    app, _saved = _make_app()

    app._restore_held_result("held", "clip", "recent-1", show_bubble=True)

    assert app._held_result == "held"
    assert app._held_clipboard == "clip"
    assert app._held_recent_result_id == "recent-1"
    assert app._overlay.held == ["held"]


def test_try_acquire_pipeline_action_returns_false_when_busy() -> None:
    app, _saved = _make_app()
    app._pipeline_lock = threading.Lock()
    app._pipeline_lock.acquire()

    acquired = app._try_acquire_pipeline_action("Ghost revert")

    assert acquired is False
    app._pipeline_lock.release()


def test_try_acquire_pipeline_action_returns_true_when_available() -> None:
    app, _saved = _make_app()
    app._pipeline_lock = threading.Lock()

    acquired = app._try_acquire_pipeline_action("Ghost revert")

    assert acquired is True
    app._pipeline_lock.release()


def test_reset_interaction_ui_uses_public_overlay_api() -> None:
    app, _saved = _make_app()
    app._hwnd_watch_active = True

    app._reset_interaction_ui(hide_hold_bubble=True, cancel_staging=True)

    assert app._hwnd_watch_active is False
    assert app._digit_interceptor.states == [False]
    assert app._overlay.hidden is True
    assert app._overlay.recording_personas_hidden is True
    assert app._overlay.realtime_preview_hidden is True
    assert app._overlay.hold_bubble_hidden is True
    assert app._overlay.staging_cancelled is True
    assert app._tray.statuses == ["Ready"]


def test_run_llm_with_watch_shows_processing_and_returns_result() -> None:
    app, _saved = _make_app()
    app._caret_x = 10
    app._caret_y = 20
    app._cancel_requested = threading.Event()
    started: list[bool] = []

    def start_watch() -> None:
        started.append(True)
        app._hwnd_watch_active = True

    app._start_hwnd_watcher = start_watch
    app._run_llm = lambda text, persona=None: "clean"

    result = app._run_llm_with_watch(
        "raw",
        cancel_before_message="before",
        cancel_during_message="during",
        cancel_after_message="after",
        error_message="error",
    )

    assert result == "clean"
    assert started == [True]
    assert app._overlay.shown == [(10, 20, "Processing...")]
    assert app._overlay.updated_statuses == []
    assert app._tray.statuses == ["Processing..."]
    assert app._hwnd_watch_active is False


def test_run_llm_with_watch_uses_corner_status_and_short_circuits_on_cancel() -> None:
    app, _saved = _make_app()
    app._cancel_requested = threading.Event()
    app._cancel_requested.set()
    resets: list[bool] = []
    llm_called = False

    def fake_reset(**_kwargs) -> None:
        resets.append(True)

    def fake_llm(text: str, persona=None) -> str:
        nonlocal llm_called
        llm_called = True
        return text

    app._reset_interaction_ui = fake_reset
    app._start_hwnd_watcher = lambda: None
    app._run_llm = fake_llm

    result = app._run_llm_with_watch(
        "raw",
        at_corner=True,
        cancel_before_message="before",
        cancel_during_message="during",
        cancel_after_message="after",
        error_message="error",
    )

    assert result is None
    assert llm_called is False
    assert resets == [True]
    assert app._overlay.shown == []
    assert app._overlay.updated_statuses == ["Processing..."]
    assert app._tray.statuses == ["Processing..."]


def test_set_last_delivery_anchor_updates_target_and_shows_ghost(monkeypatch) -> None:
    from untype import main as main_module

    app, _saved = _make_app()
    caret = SimpleNamespace(x=123, y=456)
    target = object()

    monkeypatch.setattr(main_module, "get_caret_screen_position", lambda: caret)

    app._set_last_delivery_anchor(target, show_ghost=True)

    assert app._last_target_window is target
    assert app._last_caret_x == 123
    assert app._last_caret_y == 456
    assert app._overlay.ghost_menu_positions == [(123, 456)]


def test_restore_last_interaction_context_restores_saved_fields() -> None:
    app, _saved = _make_app()
    target = object()
    app._last_mode = "polish"
    app._last_selected_text = "selection"
    app._last_original_clipboard = "clipboard"
    app._last_caret_x = 33
    app._last_caret_y = 44
    app._window_mismatch = True

    app._restore_last_interaction_context(target)

    assert app._mode == "polish"
    assert app._selected_text == "selection"
    assert app._original_clipboard == "clipboard"
    assert app._target_window is target
    assert app._caret_x == 33
    assert app._caret_y == 44
    assert app._window_mismatch is False


def test_undo_last_injection_if_possible_only_undoes_foreground_target(monkeypatch) -> None:
    from untype import main as main_module

    app, _saved = _make_app()
    target = object()
    app._last_target_window = target
    undos: list[bool] = []

    monkeypatch.setattr(
        main_module,
        "verify_foreground_window",
        lambda candidate: candidate is target,
    )
    app._simulate_undo = lambda: undos.append(True)

    returned = app._undo_last_injection_if_possible("Ghost regenerate")

    assert returned is target
    assert undos == [True]

    app._last_target_window = object()
    returned = app._undo_last_injection_if_possible("Ghost regenerate")

    assert returned is app._last_target_window
    assert undos == [True]


class FakeTimer:
    instances: list["FakeTimer"] = []

    def __init__(self, interval, function, args=None, kwargs=None) -> None:
        self.interval = interval
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.cancelled = False
        self.started = False
        self.daemon = False
        self.name: str | None = None
        self.__class__.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.function(*self.args, **self.kwargs)


def _make_config_app() -> UnTypeApp:
    app = object.__new__(UnTypeApp)
    app._config = AppConfig()
    app._config_save_lock = threading.Lock()
    app._config_save_timer = None
    app._config_save_generation = 0
    return app


def test_schedule_config_save_coalesces_to_latest_snapshot(monkeypatch) -> None:
    from untype import main as main_module

    FakeTimer.instances = []
    saved: list[str] = []
    app = _make_config_app()

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)
    monkeypatch.setattr(
        main_module,
        "save_config",
        lambda config: saved.append(config.last_selected_persona),
    )

    app._config.last_selected_persona = "first"
    app._schedule_config_save(delay=0.25, reason="persona selection")
    first = FakeTimer.instances[-1]

    app._config.last_selected_persona = "second"
    app._schedule_config_save(delay=0.25, reason="persona selection")
    second = FakeTimer.instances[-1]

    assert first.cancelled is True
    assert second.started is True

    first.fire()
    assert saved == []

    second.fire()
    assert saved == ["second"]
    assert app._config_save_timer is None


def test_flush_pending_config_save_persists_latest_snapshot(monkeypatch) -> None:
    from untype import main as main_module

    FakeTimer.instances = []
    saved: list[str] = []
    app = _make_config_app()

    monkeypatch.setattr(main_module.threading, "Timer", FakeTimer)
    monkeypatch.setattr(
        main_module,
        "save_config",
        lambda config: saved.append(config.last_selected_persona),
    )

    app._config.last_selected_persona = "pending"
    app._schedule_config_save(delay=0.25, reason="persona selection")
    timer = FakeTimer.instances[-1]

    app._flush_pending_config_save()

    assert timer.cancelled is True
    assert saved == ["pending"]
    assert app._config_save_timer is None


def test_deliver_result_holds_when_injection_fails(monkeypatch) -> None:
    from untype import main as main_module

    app, saved = _make_app()
    app._verify_window_safety = lambda: True

    monkeypatch.setattr(
        main_module,
        "inject_text",
        lambda text, clipboard: InjectionResult(True, False),
    )

    delivered = app._deliver_result("raw", "clean")

    assert delivered is False
    assert saved == [("raw", "clean", None, False)]
    assert app._held_result == "clean"
    assert app._held_clipboard == "original"
    assert app._recent_results[0].status == "held"
    assert app._overlay.held == ["clean"]
    assert app._tray.statuses == ["Ready"]


def test_recording_personas_prefers_quick_subset() -> None:
    app, _saved = _make_app()
    app._personas = [
        Persona(id="a", name="A", icon="A", active=True, quick=False),
        Persona(id="b", name="B", icon="B", active=True, quick=True),
        Persona(id="c", name="C", icon="C", active=True, quick=True),
    ]

    assert [persona.id for persona in app._recording_personas] == ["b", "c"]


def test_recording_personas_falls_back_to_first_four_active() -> None:
    app, _saved = _make_app()
    app._personas = [
        Persona(id=f"p{i}", name=f"P{i}", icon=str(i), active=True)
        for i in range(1, 6)
    ]

    assert [persona.id for persona in app._recording_personas] == ["p1", "p2", "p3", "p4"]
