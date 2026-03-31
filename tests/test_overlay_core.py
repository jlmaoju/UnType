from __future__ import annotations

import queue
import threading

from untype.overlay import CapsuleOverlay


def test_cancel_staging_enqueues_public_command() -> None:
    overlay = object.__new__(CapsuleOverlay)
    overlay._queue = queue.Queue()

    overlay.cancel_staging()

    assert overlay._queue.get_nowait() == ("STAGING_CANCEL",)


def test_do_cancel_staging_unblocks_waiter_without_widget() -> None:
    overlay = object.__new__(CapsuleOverlay)
    overlay._staging_text_widget = None
    overlay._staging_event = threading.Event()
    overlay._staging_result_text = "draft"
    overlay._staging_result_action = ""
    hidden: list[bool] = []
    overlay._do_hide_staging = lambda: hidden.append(True)

    overlay._do_cancel_staging()

    assert overlay._staging_result_text == ""
    assert overlay._staging_result_action == "cancel"
    assert overlay._staging_event.is_set() is True
    assert hidden == [True]
