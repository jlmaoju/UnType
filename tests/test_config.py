from __future__ import annotations

import json
from pathlib import Path

import pytest

from untype import config as config_module


def test_load_config_creates_default_file_when_missing(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)

    config = config_module.load_config()

    assert config.hotkey.trigger == "f6"
    assert config.stt.backend == "realtime_api"
    assert config_path.exists()


def test_dict_to_config_clamps_and_sanitizes_values() -> None:
    config = config_module._dict_to_config(
        {
            "audio": {"gain_boost": 20.0, "sample_rate": 999999},
            "stt": {"backend": "bad-backend", "api_base_url": "not-a-url"},
            "llm": {"base_url": "also-bad", "temperature": 5.0},
        }
    )

    assert config.audio.gain_boost == 10.0
    assert config.audio.sample_rate == 48000
    assert config.stt.backend == "api"
    assert config.stt.api_base_url == ""
    assert config.llm.base_url == ""
    assert config.llm.temperature == 2.0


def test_load_personas_skips_invalid_entries(monkeypatch, tmp_path: Path) -> None:
    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()

    (personas_dir / "01_valid.json").write_text(
        json.dumps({"id": "valid", "name": "Valid", "icon": "*", "active": True}),
        encoding="utf-8",
    )
    (personas_dir / "02_missing_fields.json").write_text(
        json.dumps({"id": "broken"}),
        encoding="utf-8",
    )
    (personas_dir / "03_invalid.json").write_text("{", encoding="utf-8")

    monkeypatch.setattr(config_module, "get_personas_dir", lambda: personas_dir)

    personas = config_module.load_personas()

    assert [persona.id for persona in personas] == ["valid"]


def test_load_config_recovers_from_invalid_toml(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[broken", encoding="utf-8")
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)

    config = config_module.load_config()

    assert config.hotkey.trigger == "f6"
    assert config.stt.backend == "realtime_api"
    assert config_path.exists()


def test_save_config_restores_previous_file_when_write_fails(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    original_contents = 'language = "en"\n'
    config_path.write_text(original_contents, encoding="utf-8")
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)

    def fake_dump(data, file_obj) -> None:
        file_obj.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(config_module.tomli_w, "dump", fake_dump)

    with pytest.raises(OSError, match="disk full"):
        config_module.save_config(config_module.AppConfig(language="zh"))

    assert config_path.read_text(encoding="utf-8") == original_contents
    assert not config_path.with_suffix(".toml.tmp").exists()


def test_save_and_delete_persona_round_trip(monkeypatch, tmp_path: Path) -> None:
    personas_dir = tmp_path / "personas"
    monkeypatch.setattr(config_module, "get_personas_dir", lambda: personas_dir)

    persona = config_module.Persona(id="demo", name="Demo", icon="*")

    config_module.save_persona(persona)

    assert (personas_dir / "demo.json").exists()
    assert [loaded.id for loaded in config_module.load_personas()] == ["demo"]
    assert config_module.delete_persona("demo") is True
    assert config_module.delete_persona("demo") is False
