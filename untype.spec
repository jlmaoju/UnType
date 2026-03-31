# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for the full UnType build.

This variant includes local STT support (`faster-whisper`, `ctranslate2`)
in addition to the API-based features.
"""

from pathlib import Path

project_root = Path(SPECPATH).resolve()

personas_dir = project_root / "personas"
locales_dir = project_root / "locales"

datas = []
if personas_dir.exists():
    datas.append((str(personas_dir), "personas"))
if locales_dir.exists():
    datas.append((str(locales_dir), "locales"))

hiddenimports = [
    "sounddevice",
    "_sounddevice_data",
    "pynput",
    "pynput.keyboard",
    "pynput.mouse",
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "pystray",
    "pystray._win32",
    "PIL",
    "PIL._tkinter_finder",
    "PIL.Image",
    "PIL.ImageTk",
    "PIL.PngImagePlugin",
    "win32gui",
    "win32con",
    "win32api",
    "pywintypes",
    "tomli",
    "tomli_w",
    "httpx",
    "h11",
    "anyio",
    "sniffio",
    "dashscope",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
    "faster_whisper",
    "ctranslate2",
    "ctranslate2._ext",
    "ctranslate2.libs",
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
        "matplotlib",
        "numpy.f2py",
        "pandas",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
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
    name="untype",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
    name="untype",
)
