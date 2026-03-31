from __future__ import annotations

from pathlib import Path

import build


def test_build_pyinstaller_command_uses_full_spec_by_default() -> None:
    cmd = build.build_pyinstaller_command()

    assert cmd == ["uv", "run", "pyinstaller", "untype.spec"]


def test_build_pyinstaller_command_uses_online_spec() -> None:
    cmd = build.build_pyinstaller_command(online=True)

    assert cmd == ["uv", "run", "pyinstaller", "untype-online.spec"]


def test_build_pyinstaller_command_onefile_includes_expected_assets_and_imports() -> None:
    cmd = build.build_pyinstaller_command(onefile=True)

    assert cmd[:6] == ["uv", "run", "pyinstaller", "--onefile", "--name", "untype"]
    assert "--noconsole" in cmd
    assert f"personas{build.os.pathsep}personas" in cmd
    assert f"locales{build.os.pathsep}locales" in cmd
    assert "faster_whisper" in cmd
    assert "dashscope" in cmd
    assert cmd[-1] == str(Path("src") / "untype" / "main.py")


def test_run_pyinstaller_passes_command_to_subprocess(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class Result:
        returncode = 7

    def fake_run(cmd: list[str], cwd: Path):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        return Result()

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    rc = build.run_pyinstaller(onefile=True)

    assert rc == 7
    assert seen["cmd"] == build.build_pyinstaller_command(onefile=True)
    assert seen["cwd"] == build.PROJECT_ROOT


def test_check_python_module_uses_import_name_alias(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class Result:
        returncode = 0

    def fake_run(cmd: list[str], *, capture_output: bool, timeout: int):
        seen["cmd"] = cmd
        return Result()

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    assert build.check_python_module("tomli-w", import_name="tomli_w") is True
    assert seen["cmd"][:2] == [build.sys.executable, "-c"]
    assert "importlib.import_module('tomli_w')" in seen["cmd"][2]


def test_check_dependencies_uses_runtime_import_aliases(monkeypatch) -> None:
    seen: list[tuple[str, str | None]] = []

    monkeypatch.setattr(build, "check_command_available", lambda command: True)
    monkeypatch.setattr(build, "check_module_version", lambda module, import_name=None: "1.0")

    def fake_check_python_module(module_name: str, import_name: str | None = None) -> bool:
        seen.append((module_name, import_name))
        return True

    monkeypatch.setattr(build, "check_python_module", fake_check_python_module)

    assert build.check_dependencies(verbose=False) == []
    assert ("faster-whisper", "faster_whisper") in seen
    assert ("Pillow", "PIL") in seen
    assert ("tomli-w", "tomli_w") in seen
