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

**Built-in 6 Persona Masks** for different contexts:
- 👔 To Boss — Formal, tactful workplace communication
- 🤝 To Colleague — Friendly yet professional daily exchange
- 📋 Bullet Points — Auto-organize into a concise list
- 🌐 English — Chinese speech → English output
- 🗣️ Plain Talk — Make complex ideas simple
- 🙅 Decline — Politely turn down requests

Press a digit key (1-6) during recording to switch, or right-click tray icon → **Personas** to customize.

## Core Features

- **AI-refined output** — LLM automatically fixes punctuation, filler words, grammar, and recognition errors
- **Voice-edit selected text** — Select text, speak an instruction, and the LLM applies it
- **Push-to-Talk** — Press F6 to start recording, press again to stop; works in any application
- **Triple STT backends** — Online API, local inference, Aliyun realtime streaming API

## Experience Details

- **Realtime transcription preview** — See recognized text appear during recording with Aliyun realtime API, similar to WeChat voice input
- **Volume visualization** — Real-time volume bar at the bottom of the capsule during recording
- **Ghost Menu** — Post-injection undo menu: revert to raw draft, regenerate, or reopen editor. No countdown pressure.
- **Adjustable capsule position** — Choose fixed (draggable, position saved) or follow cursor mode
- **Hotkey recording** — Click the input field in settings and press your desired key
- **System tray UI** — Color-coded status indicator + settings dialog

## Quick Start

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
- An OpenAI-compatible STT API key (for online mode), or a GPU for local Whisper inference
- An OpenAI-compatible LLM API key (for text refinement; optional but recommended)

## Configuration

Settings are stored in `~/.untype/config.toml` (created on first launch):

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `hotkey` | `trigger` | `f6` | Push-to-talk hotkey |
| `hotkey` | `mode` | `toggle` | `toggle` (press to start/stop) or `hold` (hold to speak) |
| `overlay` | `capsule_position_mode` | `"fixed"` | Capsule position mode: `"fixed"` (draggable) or `"caret"` (follow cursor) |
| `audio` | `gain_boost` | `3.0` | Gain multiplier for quiet speech |
| `stt` | `backend` | `api` | `api`, `local`, or `realtime_api` |

### STT Backend Selection

**Online API (default)**
- Uses OpenAI-compatible `/audio/transcriptions` interface
- Works with any proxy service
- Returns complete result after recording ends

**Local Model**
- Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local inference
- Requires GPU with CUDA support
- Better privacy, no internet needed

**Aliyun Realtime API (recommended)**
- Uses Aliyun DashScope realtime speech recognition
- **WebSocket streaming with live transcription preview during recording**
- **Ultra-low latency, experience similar to WeChat voice input**
- Requires [Aliyun DashScope API Key](https://dashscope.console.aliyun.com/)

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

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Changelog

### v0.2.0 (2025-02-25)
- Add Aliyun realtime speech recognition backend with live transcription preview during recording
- Add fixed capsule position mode (draggable, position persisted)
- Add settings UI dynamic field visibility (show/hide based on backend selection)
- Fix hotkey listener restart race condition
- Add hotkey blacklist to prevent system shortcut conflicts
- Fix ghost menu position to follow capsule configuration
