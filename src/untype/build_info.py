"""Build-time feature flags.

This module detects which features are available at runtime,
based on whether optional dependencies were bundled during the build.
"""

try:
    import faster_whisper

    HAS_LOCAL_STT = True
except ImportError:
    HAS_LOCAL_STT = False
