#!/usr/bin/env python
"""Build script for UnType — wraps PyInstaller with sensible defaults.

Usage:
    python build.py              # Build with PyInstaller
    python build.py --clean      # Clean build artifacts first
    python build.py --onefile    # Build as single .exe (larger, slower startup)
    python build.py --check-deps # Check dependencies only
"""

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
FULL_SPEC = PROJECT_ROOT / "untype.spec"
ONLINE_SPEC = PROJECT_ROOT / "untype-online.spec"
ENTRY_SCRIPT = PROJECT_ROOT / "src" / "untype" / "main.py"

# Dependencies required for building
BUILD_DEPENDENCIES = {
    "pyinstaller": ("PyInstaller", "6.0.0"),
    "uv": ("uv", "0.1.0"),
}

# Runtime dependencies to verify (optional but helpful).
# Keys are package names, values are the import paths used to verify them.
RUNTIME_DEPENDENCIES = {
    "faster-whisper": "faster_whisper",
    "sounddevice": "sounddevice",
    "numpy": "numpy",
    "pynput": "pynput",
    "pyperclip": "pyperclip",
    "httpx": "httpx",
    "pystray": "pystray",
    "Pillow": "PIL",
    "tomli-w": "tomli_w",
    "dashscope": "dashscope",
}

# Hidden imports needed by the generated PyInstaller CLI builds.
# The spec files remain the source of truth for the packaged variants, but
# onefile builds rely on these explicit imports.
COMMON_HIDDEN_IMPORTS = [
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
]
FULL_BUILD_HIDDEN_IMPORTS = [
    *COMMON_HIDDEN_IMPORTS,
    "faster_whisper",
    "ctranslate2",
    "ctranslate2._ext",
    "ctranslate2.libs",
]


class BuildError(Exception):
    """Base exception for build errors."""

    pass


class DependencyError(BuildError):
    """Exception raised when a required dependency is missing."""

    pass


def get_version() -> str | None:
    """Extract version from pyproject.toml.

    Returns:
        The version string from pyproject.toml, or None if not found.
    """
    try:
        with open(PYPROJECT_TOML, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version")
    except FileNotFoundError:
        print(f"Warning: {PYPROJECT_TOML} not found", file=sys.stderr)
        return None
    except tomllib.TOMLDecodeError as e:
        print(f"Warning: Failed to parse {PYPROJECT_TOML}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Unexpected error reading version: {e}", file=sys.stderr)
        return None


def check_command_available(command: str) -> bool:
    """Check if a command is available on the system PATH.

    Args:
        command: The command to check.

    Returns:
        True if the command is available, False otherwise.
    """
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            shell=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_python_module(module_name: str, import_name: str | None = None) -> bool:
    """Check if a Python module can be imported.

    Args:
        module_name: The module name to check.
        import_name: Optional import path when it differs from the package name.

    Returns:
        True if the module is available, False otherwise.
    """
    if import_name is None:
        import_name = module_name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import importlib; importlib.import_module({import_name!r})",
            ],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except subprocess.SubprocessError:
        return False


def check_module_version(module_name: str, import_name: str | None = None) -> str | None:
    """Get the version of an installed Python module.

    Args:
        module_name: The pip package name.
        import_name: The import name (defaults to module_name).

    Returns:
        The version string, or None if the module is not installed.
    """
    if import_name is None:
        import_name = module_name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import importlib; print(importlib.import_module({import_name!r}).__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except subprocess.SubprocessError:
        pass

    # Fallback: try pip show
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", module_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except subprocess.SubprocessError:
        pass

    return None


def check_dependencies(verbose: bool = True) -> list[str]:
    """Check if all required build dependencies are available.

    Args:
        verbose: If True, print detailed status messages.

    Returns:
        A list of missing dependency names.

    Raises:
        DependencyError: If critical dependencies are missing.
    """
    if verbose:
        print("Checking build dependencies...\n")

    missing = []
    # Check Python version
    if verbose:
        print(f"Python version: {sys.version}")

    min_python = (3, 11)
    if sys.version_info < min_python:
        missing.append(f"Python {min_python[0]}.{min_python[1]}+")
        if verbose:
            print(
                "  [X] Python "
                f"{min_python[0]}.{min_python[1]}+ required "
                f"(current: {sys.version_info[0]}.{sys.version_info[1]})"
            )
    elif verbose:
        print("  [OK] Python version OK\n")

    # Check build tools
    for dep, (import_name, min_version) in BUILD_DEPENDENCIES.items():
        if dep == "uv":
            # uv is a command-line tool
            available = check_command_available("uv")
            if available:
                if verbose:
                    print("  [OK] uv (command-line tool)")
            else:
                missing.append(dep)
                if verbose:
                    print("  [X] uv not found - install from https://github.com/astral-sh/uv")
        else:
            # PyInstaller is a Python module
            version = check_module_version(dep, import_name)
            if version:
                if verbose:
                    print(f"  [OK] {dep} {version}")
            else:
                missing.append(dep)
                if verbose:
                    print(f"  [X] {dep} not installed")

    # Check runtime dependencies (optional but helpful to detect issues early)
    if verbose:
        print("\nChecking runtime dependencies...")

    runtime_missing = []
    for dep, import_name in RUNTIME_DEPENDENCIES.items():
        available = check_python_module(dep, import_name=import_name)
        if available:
            if verbose:
                print(f"  [OK] {dep}")
        else:
            runtime_missing.append(dep)
            if verbose:
                print(f"  [X] {dep} not found")

    if runtime_missing:
        if verbose:
            print("\n[!] Some runtime dependencies are missing.")
            print("  Run 'uv sync' to install all dependencies.")
        print()
    elif verbose:
        print()

    if missing:
        msg = f"Missing required build dependencies: {', '.join(missing)}"
        if verbose:
            print(f"\n[ERROR] {msg}\n")
        raise DependencyError(msg)

    if verbose:
        print("[OK] All build dependencies satisfied.\n")

    return runtime_missing


def clean() -> None:
    """Remove build artifacts."""
    print("Cleaning build artifacts...")
    try:
        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)
            print(f"  Removed {BUILD_DIR}")
        if DIST_DIR.exists():
            shutil.rmtree(DIST_DIR)
            print(f"  Removed {DIST_DIR}")

        # Also remove __pycache__
        for pycache in PROJECT_ROOT.rglob("__pycache__"):
            try:
                shutil.rmtree(pycache)
                print(f"  Removed {pycache}")
            except OSError:
                pass  # May be deleted already

        # Remove .pyc files
        for pyc in PROJECT_ROOT.rglob("*.pyc"):
            try:
                pyc.unlink()
                print(f"  Removed {pyc}")
            except OSError:
                pass  # May be deleted already

    except Exception as e:
        raise BuildError(f"Failed to clean build artifacts: {e}")


def _add_data_args(*directories: str) -> list[str]:
    """Build ``--add-data`` arguments for PyInstaller CLI usage."""
    args: list[str] = []
    for directory in directories:
        args.extend(["--add-data", f"{directory}{os.pathsep}{directory}"])
    return args


def _hidden_import_args(imports: list[str]) -> list[str]:
    """Build repeated ``--hidden-import`` arguments."""
    args: list[str] = []
    for import_name in imports:
        args.extend(["--hidden-import", import_name])
    return args


def build_pyinstaller_command(onefile: bool = False, online: bool = False) -> list[str]:
    """Build the PyInstaller command for the requested output variant."""
    cmd = ["uv", "run", "pyinstaller"]

    if online:
        cmd.append(ONLINE_SPEC.name)
        return cmd

    if onefile:
        cmd.extend(
            [
                "--onefile",
                "--name",
                "untype",
                "--noconsole",
                *_add_data_args("personas", "locales"),
                *_hidden_import_args(FULL_BUILD_HIDDEN_IMPORTS),
                str(ENTRY_SCRIPT.relative_to(PROJECT_ROOT)),
            ]
        )
        return cmd

    cmd.append(FULL_SPEC.name)
    return cmd


def run_pyinstaller(onefile: bool = False, online: bool = False) -> int:
    """Run PyInstaller with the spec file.

    Args:
        onefile: If True, build as a single .exe file.
        online: If True, build the online version (excludes local STT).

    Returns:
        The return code from PyInstaller.

    Raises:
        BuildError: If PyInstaller fails.
    """
    cmd = build_pyinstaller_command(onefile=onefile, online=online)

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode
    except KeyboardInterrupt:
        print("\nBuild interrupted by user.")
        return 130
    except Exception as e:
        raise BuildError(f"PyInstaller failed: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build UnType with PyInstaller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py              # Build with PyInstaller
  python build.py --clean      # Clean build artifacts first
  python build.py --onefile    # Build as single .exe (larger, slower startup)
  python build.py --online     # Build online version (excludes local STT)
  python build.py --check-deps # Check dependencies only
        """,
    )
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    parser.add_argument(
        "--onefile",
        action="store_true",
        help="Build as single .exe (experimental)",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Build online version (excludes local STT)",
    )
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies and exit")
    parser.add_argument("--no-deps-check", action="store_true", help="Skip dependency checking")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    try:
        # Get version for display
        version = get_version()
        version_str = f" v{version}" if version else ""

        # Check dependencies unless explicitly skipped
        if not args.no_deps_check:
            try:
                check_dependencies(verbose=True)
            except DependencyError as e:
                print(f"Error: {e}")
                return 1

        # If only checking dependencies, exit now
        if args.check_deps:
            print("Dependency check complete.")
            return 0

        if args.clean:
            clean()

        print(f"\n{'='*50}")
        print(f"Building UnType{version_str} with PyInstaller")
        print(f"{'='*50}\n")

        rc = run_pyinstaller(onefile=args.onefile, online=args.online)

        if rc == 0:
            if args.online:
                exe_path = DIST_DIR / "untype-online" / "untype-online.exe"
            elif args.onefile:
                exe_path = DIST_DIR / "untype.exe"
            else:
                exe_path = DIST_DIR / "untype" / "untype.exe"

            print(f"\n{'='*50}")
            print("Build successful!")
            print(f"Output: {exe_path}")

            # Calculate size
            if exe_path.exists():
                if args.onefile:
                    size = exe_path.stat().st_size
                    print(f"Size: {size / 1024 / 1024:.1f} MB")
                else:
                    folder_name = "untype-online" if args.online else "untype"
                    total_size = sum(
                        f.stat().st_size
                        for f in (DIST_DIR / folder_name).rglob("*")
                        if f.is_file()
                    )
                    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")

            print(f"{'='*50}\n")
        else:
            print(f"\n{'='*50}")
            print(f"Build failed with exit code {rc}")
            print(f"{'='*50}\n")

        return rc

    except BuildError as e:
        print(f"Build error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nBuild interrupted by user.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
