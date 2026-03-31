from __future__ import annotations

import builtins

from untype import build_info


def test_has_module_returns_true_when_import_succeeds(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "demo_optional_module":
            return object()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert build_info._has_module("demo_optional_module") is True


def test_has_module_returns_false_when_import_fails(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "missing_optional_module":
            raise ImportError("missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert build_info._has_module("missing_optional_module") is False
