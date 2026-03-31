# UnType (忘言)

> The fish trap exists because of the fish; once you've gotten the fish, you can forget the trap. The rabbit snare exists because of the rabbit; once you've gotten the rabbit, you can forget the snare. Words exist because of meaning; once you've gotten the meaning, you can forget the words.
> — *Zhuangzi, "External Things"*

[中文](README.md)

**UnType** is an open-source, AI-powered voice input tool for Windows. It doesn't just transcribe — it **thinks**. One hotkey, two superpowers:

1. **Speak to insert** — Your speech is transcribed by STT, then an LLM automatically refines it into clean text: removing filler words ("um", "uh", "嗯", "那个"), fixing punctuation, correcting recognition errors. What reaches your cursor is a polished draft, not a raw dump.

2. **Select to polish** — Select existing text, speak an instruction ("make it shorter", "translate to English", "rewrite in a formal tone"), and the LLM rewrites it for you.

## Why UnType?

Most voice input tools give you raw transcription — full of filler words, broken punctuation, and recognition errors. You end up spending time fixing what was supposed to save you time.

**UnType = STT + LLM.** Your speech is transcribed, then an LLM refines it into clean, well-formatted text — ready to use as-is.

**Built-in 8 Persona Masks** for different contexts:
- ✨ Default — Regular polish style, clean and natural
- 🌙 Poetic — Ornate literary style with metaphors and refined vocabulary
- 👔 To Boss — Formal, tactful workplace communication
- 🤝 To Colleague — Friendly yet professional daily exchange
- 📋 Bullet Points — Auto-organize into a concise list
- 🌐 English — Chinese speech → English output
- 🗣️ Plain Talk — Make complex ideas simple
- 🙅 Decline — Politely turn down requests

Press a digit key (1-9) during recording to switch. Your choice is remembered for next time. Only **active** personas are shown during recording — manage activation in **Personas** dialog.

## Core Features

- **AI-refined output** — LLM automatically fixes punctuation, filler words, grammar, and recognition errors
- **Voice-edit selected text** — Select text, speak an instruction, and the LLM applies it
- **Push-to-Talk** — Press F6 to start recording, press again to stop; works in any application
- **Triple STT backends** — Online API, local inference, Aliyun realtime streaming API

## Experience Details

- **Realtime transcription preview** — See recognized text appear during recording with Aliyun realtime API, similar to WeChat voice input
- **Recording duration display** — Shows elapsed time on capsule (e.g., "1:23"), auto-stops after 5 minutes
- **Volume visualization** — Real-time volume bar at the bottom of the capsule during recording
- **Persona memory** — Remembers your last selected persona, auto-selects it next time
- **Persona activation** — Enable/disable personas to control which appear during recording
- **First-run wizard** — Guided setup for new users to configure STT and LLM APIs
- **Ghost Menu** — Post-injection undo menu: revert to raw draft, regenerate, or reopen editor. No countdown pressure.
- **Adjustable capsule position** — Choose fixed (draggable, position saved) or follow cursor mode
- **Hotkey recording** — Click the input field in settings and press your desired key
- **File logging** — Open logs folder from Settings for troubleshooting

## Quick Start

<p align="center">
  <a href="https://github.com/jlmaoju/UnType/releases">
    <img src="https://img.shields.io/github/v/release/jlmaoju/UnType?style=for-the-badge&logo=windows&label=Download&color=0066CC" alt="Download">
  </a>
</p>

### 📥 Download Pre-built Version (Recommended)

Don't want to install Python? Download the `.exe` from [Releases](https://github.com/jlmaoju/UnType/releases) and double-click to run.

| Version | Size | Description |
|---------|------|-------------|
| **Full** | ~275MB | All features, including local Whisper model support |
| **Online** | ~90MB | API-only mode, smaller size, suitable for users who only need online services |

> 💡 **Note**: The Online version doesn't support local STT models, but all other features are identical. If you don't need offline speech recognition, the Online version is recommended.

### 💻 Build from Source

```bash
git clone https://github.com/jlmaoju/UnType.git
cd untype
uv sync
uv run untype
```

1. A green circle appears in the system tray. Right-click → **Settings** → fill in your API keys.
2. Click in any text field, press **F6** once to start recording, speak, press **F6** again to stop.
3. Polished text appears at your cursor.

## Requirements

- Windows 10/11
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- A working microphone
- An Aliyun DashScope API key (default realtime mode), or an OpenAI-compatible STT API key (for `api` mode), or a GPU for local Whisper inference
- An OpenAI-compatible LLM API key (for text refinement; optional but recommended)

## Configuration

Settings are stored in `~/.untype/config.toml` (created on first launch):

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `hotkey` | `trigger` | `f6` | Push-to-talk hotkey |
| `hotkey` | `mode` | `toggle` | `toggle` (press to start/stop) or `hold` (hold to speak) |
| `overlay` | `capsule_position_mode` | `"fixed"` | Capsule position mode: `"fixed"` (draggable) or `"caret"` (follow cursor) |
| `audio` | `gain_boost` | `1.5` | Gain multiplier for quiet speech |
| `stt` | `backend` | `realtime_api` | `realtime_api` (Aliyun), `api`, or `local` |

### STT Backend Selection

**Aliyun Realtime API (default, recommended)**
- Uses Aliyun DashScope realtime speech recognition
- **WebSocket streaming with live transcription preview during recording**
- **Ultra-low latency, experience similar to WeChat voice input**
- Requires [Aliyun DashScope API Key](https://dashscope.console.aliyun.com/)

**OpenAI-compatible API**
- Uses OpenAI-compatible `/audio/transcriptions` interface
- Works with any proxy service
- Returns complete result after recording ends

**Local Model**
- Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local inference
- Requires GPU with CUDA support
- Better privacy, no internet needed

## How It Works

```
Press hotkey once → Speak → Press hotkey again to stop
                ↓
   (During recording: persona bar visible,
    press 1-9 to pre-select a persona)
                ↓
        [ STT: speech → raw text ]
                ↓
   ┌─── Personas configured? ───┐
   │ YES                        │ NO
   ↓                            ↓
[ LLM: with persona ]   [ Staging area: edit ]
   ↓                            ↓
Text appears at cursor ✓  [ LLM → cursor ✓ ]
                ↓
       (Ghost menu appears)
```

**Two modes, auto-detected:**

| Mode | Trigger | What happens |
|------|---------|-------------|
| **Insert** | No text selected | Speech → STT → LLM cleanup → insert at cursor |
| **Polish** | Text selected | Speech becomes an instruction → LLM modifies the selected text |

## Development

```bash
uv run ruff check src/      # Lint
uv run ruff format src/      # Format
uv run pytest                # Run tests
```

The repository also includes a GitHub Actions CI workflow for Windows that
runs lint and tests on Python 3.11 and 3.13.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Changelog

### v0.4.0 (2026-03-31)
- **Add Recent Results panel** - Access the latest voice results from the tray and quickly copy, re-inject, or reopen them in the editor
- **Simplify persona selection during recording** - Support quick personas so the recording bar focuses on a few high-frequency choices instead of the full list
- **Restore key preferences correctly** - Language and last selected persona are now loaded back from config on startup
- **Improve hotkey recording for punctuation and keypad keys** - Prevent spurious `Alt` capture and accept aliases such as keypad minus
- **Switch text output to typed injection** - Output is now typed character-by-character instead of pasted from the clipboard
- **Harden runtime delivery logic** - Better handling for realtime STT startup failures, delivery fallbacks, ghost actions, and configuration save debounce
- **Expand regression coverage** - Add tests for build, config, hotkey, i18n, LLM, clipboard, overlay, and main runtime helpers

### v0.3.0 (2025-02-28)
- **Add first-run setup wizard** — Guided configuration for new users, with STT/LLM API setup and connection testing
- **Add persona activation feature** — Select personas to activate during wizard; enable/disable in persona manager
- **Add grid-based persona selection** — 3×3 card layout in wizard, click to toggle activation
- **Add rerun wizard button** — "Rerun Setup Wizard" button in settings dialog for reconfiguration
- **Add LLM connection verification** — Test API connection directly in the wizard and settings
- **Improve onboarding experience** — Real-time configuration preview, API validation, streaming API renamed
- **Update translations** — Add persona activation-related strings in Chinese and English
- **Polish wizard UI** — Dark theme, Zhuangzi quote, improved card descriptions

### v0.2.1 (2025-02-26)
- Add "Default" persona (regular polish style)
- Add "Poetic" persona (ornate literary style with metaphors)
- Add persona memory feature — remembers your last selected persona
- Add recording duration display (shows time like "1:23" on capsule)
- Add recording timeout protection (auto-stops after 5 minutes)
- Add file logging (open log folder from Settings)
- Change default STT backend to Aliyun Realtime API
- Adjust default audio gain to 1.5
- Move "Open Logs" button to Settings dialog

### v0.2.0 (2025-02-25)
- Add Aliyun realtime speech recognition backend with live transcription preview during recording
- Add fixed capsule position mode (draggable, position persisted)
- Add settings UI dynamic field visibility (show/hide based on backend selection)
- Fix hotkey listener restart race condition
- Add hotkey blacklist to prevent system shortcut conflicts
- Fix ghost menu position to follow capsule configuration
