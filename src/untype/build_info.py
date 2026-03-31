"""Build-time feature flags.

This module detects which features are available at runtime,
based on whether optional dependencies were bundled during the build.
"""

def _has_module(module_name: str) -> bool:
    """Return whether *module_name* can be imported."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


HAS_LOCAL_STT = _has_module("faster_whisper")
