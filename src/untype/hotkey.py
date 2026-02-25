"""Global hotkey listener using pynput with push-to-talk semantics."""

import logging
import threading
from typing import Callable

from pynput import keyboard

logger = logging.getLogger(__name__)

# Map modifier name -> set of pynput Key variants (left/right)
_MODIFIER_MAP: dict[str, set[keyboard.Key]] = {
    "alt": {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr},
    "ctrl": {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r},
    "shift": {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r},
    "cmd": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
    "win": {keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r},
}

# Reverse lookup: any pynput Key variant -> canonical modifier name
_KEY_TO_MODIFIER: dict[keyboard.Key, str] = {}
for _name, _variants in _MODIFIER_MAP.items():
    if _name == "win":
        continue  # "win" is an alias for "cmd", skip to avoid overwrite
    for _key in _variants:
        _KEY_TO_MODIFIER[_key] = _name

# Named special keys that aren't modifiers
_SPECIAL_KEYS: dict[str, keyboard.Key] = {
    "space": keyboard.Key.space,
    "enter": keyboard.Key.enter,
    "tab": keyboard.Key.tab,
    "backspace": keyboard.Key.backspace,
    "delete": keyboard.Key.delete,
    "escape": keyboard.Key.esc,
    "esc": keyboard.Key.esc,
    "up": keyboard.Key.up,
    "down": keyboard.Key.down,
    "left": keyboard.Key.left,
    "right": keyboard.Key.right,
    "home": keyboard.Key.home,
    "end": keyboard.Key.end,
    "page_up": keyboard.Key.page_up,
    "page_down": keyboard.Key.page_down,
    "insert": keyboard.Key.insert,
    "f1": keyboard.Key.f1,
    "f2": keyboard.Key.f2,
    "f3": keyboard.Key.f3,
    "f4": keyboard.Key.f4,
    "f5": keyboard.Key.f5,
    "f6": keyboard.Key.f6,
    "f7": keyboard.Key.f7,
    "f8": keyboard.Key.f8,
    "f9": keyboard.Key.f9,
    "f10": keyboard.Key.f10,
    "f11": keyboard.Key.f11,
    "f12": keyboard.Key.f12,
}


def parse_hotkey(hotkey_str: str) -> tuple[set[str], keyboard.KeyCode | keyboard.Key]:
    """Parse a hotkey string like "alt+space" into (modifier_names, trigger_key).

    The last component is always the trigger key. All preceding components
    must be recognized modifiers.

    Args:
        hotkey_str: Combo string such as "alt+space" or "ctrl+shift+a".

    Returns:
        A tuple of (set of canonical modifier names, trigger key).

    Raises:
        ValueError: If the string is empty, has an unknown modifier, or has
            no trigger key.
    """
    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    if not parts or parts == [""]:
        raise ValueError(f"Empty hotkey string: {hotkey_str!r}")

    *modifier_parts, trigger_part = parts

    # Validate modifiers
    modifiers: set[str] = set()
    for mod in modifier_parts:
        canonical = "cmd" if mod == "win" else mod
        if canonical not in _MODIFIER_MAP:
            raise ValueError(
                f"Unknown modifier {mod!r} in hotkey {hotkey_str!r}. "
                f"Supported modifiers: alt, ctrl, shift, cmd/win"
            )
        modifiers.add(canonical)

    # Resolve trigger key
    trigger: keyboard.KeyCode | keyboard.Key
    if trigger_part in _SPECIAL_KEYS:
        trigger = _SPECIAL_KEYS[trigger_part]
    elif trigger_part in _MODIFIER_MAP or trigger_part == "win":
        raise ValueError(
            f"Trigger key {trigger_part!r} is a modifier. "
            f"The last component of {hotkey_str!r} must be a non-modifier key."
        )
    elif len(trigger_part) == 1:
        trigger = keyboard.KeyCode.from_char(trigger_part)
    else:
        raise ValueError(
            f"Unknown trigger key {trigger_part!r} in hotkey {hotkey_str!r}"
        )

    return modifiers, trigger


class HotkeyListener:
    """Global hotkey listener supporting push-to-talk and toggle modes.

    In **hold** mode (push-to-talk): pressing the hotkey calls *on_press*,
    releasing it calls *on_release*.

    In **toggle** mode: the first press calls *on_press*, the second press
    calls *on_release*.  Key release events are ignored.
    """

    def __init__(
        self,
        hotkey_str: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        mode: str = "toggle",
    ) -> None:
        self._modifiers, self._trigger = parse_hotkey(hotkey_str)
        self._on_press = on_press
        self._on_release = on_release
        self._hotkey_str = hotkey_str
        self._mode = mode  # "hold" or "toggle"

        # Currently held modifier names (canonical)
        self._held_modifiers: set[str] = set()
        # Whether the trigger key is currently held
        self._trigger_held: bool = False
        # Whether the hotkey is currently active (pressed and not yet released)
        self._active: bool = False
        # Toggle mode: whether recording is currently in progress
        self._toggled_on: bool = False

        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

        logger.debug("HotkeyListener configured for %r (mode=%s)", hotkey_str, mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening for the hotkey in a daemon background thread."""
        if self._listener is not None:
            logger.warning("Listener already running")
            return

        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("Hotkey listener started for %r", self._hotkey_str)

    def stop(self) -> None:
        """Stop the hotkey listener and reset state."""
        if self._listener is not None:
            self._listener.stop()
            # Wait for the listener thread to actually stop.
            # pynput.Listener uses a private _thread attribute.
            thread = getattr(self._listener, "_thread", None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
            self._listener = None
            logger.info("Hotkey listener stopped")
        with self._lock:
            self._held_modifiers.clear()
            self._trigger_held = False
            self._active = False
            self._toggled_on = False

    def reset_toggle(self) -> None:
        """Reset the toggle state so the next press is treated as a new start.

        Call this when the pipeline is cancelled externally (e.g. by the
        emergency stop button) so that the HotkeyListener doesn't think
        it's still in the "toggled on" state.
        """
        with self._lock:
            self._toggled_on = False
            self._active = False

    # ------------------------------------------------------------------
    # Internal key handlers
    # ------------------------------------------------------------------

    def _normalize_modifier(self, key: keyboard.Key | keyboard.KeyCode) -> str | None:
        """Return the canonical modifier name if *key* is a modifier, else None."""
        if isinstance(key, keyboard.Key):
            return _KEY_TO_MODIFIER.get(key)
        return None

    def _is_trigger(self, key: keyboard.Key | keyboard.KeyCode) -> bool:
        """Return True if *key* matches the configured trigger key."""
        if isinstance(self._trigger, keyboard.Key):
            return key == self._trigger
        # KeyCode comparison — match by char (case-insensitive)
        if isinstance(key, keyboard.KeyCode) and isinstance(self._trigger, keyboard.KeyCode):
            if key.char is not None and self._trigger.char is not None:
                return key.char.lower() == self._trigger.char.lower()
        return False

    def _on_key_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        fire_press = False
        fire_release = False

        with self._lock:
            mod = self._normalize_modifier(key)
            if mod is not None:
                self._held_modifiers.add(mod)

            if self._is_trigger(key):
                self._trigger_held = True

            # Check if the full hotkey combo is pressed.
            combo_pressed = (
                self._trigger_held
                and self._modifiers <= self._held_modifiers
            )

            if self._mode == "toggle":
                # Toggle mode: first press → on_press, second press → on_release.
                # We only act on the initial combo press (not auto-repeat).
                if combo_pressed and not self._active:
                    self._active = True  # track physical key state
                    if not self._toggled_on:
                        self._toggled_on = True
                        fire_press = True
                    else:
                        self._toggled_on = False
                        fire_release = True
            else:
                # Hold mode: press → on_press (release handled in _on_key_release).
                if combo_pressed and not self._active:
                    self._active = True
                    fire_press = True

        # Fire callbacks outside the lock to avoid deadlocks.
        if fire_press:
            logger.debug("Hotkey activated: %s", self._hotkey_str)
            try:
                self._on_press()
            except Exception:
                logger.exception("Error in on_press callback")

        if fire_release:
            logger.debug("Hotkey toggled off: %s", self._hotkey_str)
            try:
                self._on_release()
            except Exception:
                logger.exception("Error in on_release callback")

    def _on_key_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        fire_release = False

        with self._lock:
            if self._mode == "toggle":
                # Toggle mode: key release never fires on_release — only
                # reset the physical key tracking so the next press is detected.
                mod = self._normalize_modifier(key)
                if self._is_trigger(key):
                    self._active = False
                    self._trigger_held = False
                if mod is not None:
                    self._held_modifiers.discard(mod)
            else:
                # Hold mode: release fires on_release if active.
                if self._active:
                    mod = self._normalize_modifier(key)
                    if self._is_trigger(key) or (mod is not None and mod in self._modifiers):
                        fire_release = True
                        self._active = False
                        logger.debug("Hotkey deactivated: %s", self._hotkey_str)

                # Update tracking state *after* deactivation check.
                mod = self._normalize_modifier(key)
                if mod is not None:
                    self._held_modifiers.discard(mod)

                if self._is_trigger(key):
                    self._trigger_held = False

        # Fire callback outside the lock to avoid deadlocks.
        if fire_release:
            try:
                self._on_release()
            except Exception:
                logger.exception("Error in on_release callback")
