# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for UnType Online Version.

This spec file builds the "online" version of UnType, which excludes
local STT dependencies (faster-whisper, ctranslate2) for a smaller
distribution size.

Build command:
    pyinstaller untype-online.spec

Output:
    dist/untype-online/untype-online.exe (onedir mode)
"""

import sys
from pathlib import Path

# Project root (where this spec file lives)
# SPECPATH is a string in PyInstaller, convert to Path
project_root = Path(SPECPATH).resolve()

# Collect all data files
personas_dir = project_root / "personas"
locales_dir = project_root / "locales"

datas = []
if personas_dir.exists():
    datas.append((str(personas_dir), "personas"))
if locales_dir.exists():
    datas.append((str(locales_dir), "locales"))

# Hidden imports that PyInstaller might miss
# NOTE: faster-whisper and ctranslate2 are NOT included in online version
hiddenimports = [
    # sounddevice / portaudio
    "sounddevice",
    "_sounddevice_data",
    # pynput
    "pynput",
    "pynput.keyboard",
    "pynput.mouse",
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    # pystray
    "pystray",
    "pystray._win32",
    # Pillow / PIL
    "PIL",
    "PIL._tkinter_finder",
    "PIL.Image",
    "PIL.ImageTk",
    "PIL.PngImagePlugin",
    # pywin32 (Windows-specific)
    "win32gui",
    "win32con",
    "win32api",
    "pywintypes",
    # tomli (Python < 3.11)
    "tomli",
    "tomli_w",
    # httpx
    "httpx",
    "h11",
    "anyio",
    "sniffio",
    # dashscope (for realtime API)
    "dashscope",
    # tkinter (usually auto-detected, but be safe)
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
]

a = Analysis(
    ["src/untype/main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unused modules to reduce size
        "matplotlib",
        "numpy.f2py",
        "pandas",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
        # Exclude local STT dependencies (key difference from full version)
        "faster-whisper",
        "faster_whisper",
        "ctranslate2",
        "ctranslate2.libs",
        # onnxruntime is a dependency of faster-whisper, not needed for online version
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        # hf_transfer is pulled by faster-whisper dependencies
        "hf_xet",
        "hf_transfer",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="untype-online",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Enable UPX compression if available
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Application icon (set to path to .ico file when available)
    # Example: icon=str(project_root / "assets" / "icon.ico")
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="untype-online",
)
