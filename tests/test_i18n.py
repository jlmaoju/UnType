from __future__ import annotations

import json
from pathlib import Path

from untype import i18n


def test_list_available_locales_returns_sorted_codes(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "zh.json").write_text("{}", encoding="utf-8")
    (tmp_path / "en.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(i18n, "get_locales_dir", lambda: tmp_path)

    assert i18n.list_available_locales() == ["en", "zh"]


def test_get_locale_display_name_uses_meta_name(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "custom.json").write_text(
        json.dumps({"meta": {"name": "Custom Language"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(i18n, "get_locales_dir", lambda: tmp_path)

    assert i18n.get_locale_display_name("custom") == "Custom Language"


def test_init_language_falls_back_to_builtin_when_no_locales(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(i18n, "get_locales_dir", lambda: tmp_path)
    monkeypatch.setattr(i18n, "_translations", {})
    monkeypatch.setattr(i18n, "_current_lang", "zh")

    i18n.init_language("missing")

    assert i18n.get_language() == "en"
    assert i18n.t("tray.settings") == "Settings..."


def test_t_uses_current_translation_then_default_then_key(monkeypatch) -> None:
    monkeypatch.setattr(i18n, "_translations", {"greeting": "Hello, {name}!"})

    assert i18n.t("greeting", name="UnType") == "Hello, UnType!"
    assert i18n.t("missing", "Fallback") == "Fallback"
    assert i18n.t("still_missing") == "still_missing"
