# UnType (忘言)

> The fish trap exists because of the fish; once you've gotten the fish, you can forget the trap. The rabbit snare exists because of the rabbit; once you've gotten the rabbit, you can forget the snare. Words exist because of meaning; once you've gotten the meaning, you can forget the words.
> — *Zhuangzi, "External Things"*

[中文](README.md)

**UnType** is an open-source, AI-powered voice input tool for Windows. It doesn't just transcribe — it **thinks**. One hotkey, two superpowers:

1. **Speak to insert** — Your speech is transcribed by STT, then an LLM automatically refines it into clean text: removing filler words ("um", "uh", "嗯", "那个"), fixing punctuation, correcting recognition errors. What reaches your cursor is a polished draft, not a raw dump.

2. **Select to polish** — Select existing text, speak an instruction ("make it shorter", "translate to English", "rewrite in a formal tone"), and the LLM rewrites it for you. Voice-controlled text editing, anywhere.

## Why UnType?

Most voice input tools give you raw transcription — full of filler words, broken punctuation, and recognition errors. You end up spending time fixing what was supposed to save you time.

**UnType = STT + LLM.** Your speech is transcribed, then an LLM refines it into clean, well-formatted text — ready to use as-is. And when you need to edit existing text, just select it and speak.

## Features

- **AI-refined output** — Not raw transcription. LLM automatically fixes punctuation, filler words, grammar, and recognition errors before text reaches your cursor.
- **Voice-edit selected text** — Select text, speak an instruction, and the LLM applies it. Like a voice-controlled find-and-replace on steroids.
- **Push-to-Talk** — Press the hotkey (default: F6) once to start recording, press again to stop. Hold mode also available. Works in any application.
- **Dual STT backends** — Online API (OpenAI-compatible) or local inference via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Your choice.
- **System tray UI** — Color-coded status indicator + settings dialog.
- **Persona Masks** — Pre-select an LLM personality during recording with a single digit key (1-9) or a click. Define custom tone profiles for different contexts: academic, workplace, casual, bullet-point notes — each with its own prompt, model, and temperature. Drop a JSON file into `personas/` to add a new persona. When a persona is pre-selected, the staging area is skipped for a faster workflow.
- **Ghost Menu** — Post-injection undo menu: revert to raw draft, regenerate, or reopen editor. No countdown pressure.
- **Adjustable capsule position** — Choose whether the capsule follows the cursor, stays at bottom center, or bottom left of the screen.

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

## Quick Start

```bash
git clone https://github.com/jlmaoju/untype.git
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
| `overlay` | `capsule_position` | `caret` | Capsule position: `caret` (follow cursor), `bottom_center`, or `bottom_left` |
| `audio` | `gain_boost` | `3.0` | Gain multiplier for quiet speech |
| `stt` | `backend` | `api` | `api` or `local` |
| `stt` | `api_base_url` | `""` | OpenAI-compatible STT API endpoint |
| `stt` | `api_key` | `""` | STT API key |
| `stt` | `api_model` | `gpt-4o-transcribe` | STT model name |
| `stt` | `model_size` | `small` | Local Whisper model size |
| `llm` | `base_url` | `""` | OpenAI-compatible chat API endpoint |
| `llm` | `api_key` | `""` | LLM API key |
| `llm` | `model` | `""` | LLM model name |

### Personas

Personas let the AI process your speech in different "roles". Pre-select a persona during recording by pressing a digit key (1-9) or clicking, then skip the staging area and go straight to the LLM.

**Built-in Personas:**

| Icon | Name | Use Case |
|------|------|----------|
| 👔 | To Boss | Formal, tactful workplace communication |
| 🤝 | To Colleague | Friendly yet professional daily exchange |
| 📋 | Bullet Points | Auto-organize into a concise list |
| 🌐 | English | Chinese speech → English output |
| 🗣️ | Plain Talk | Make complex ideas simple |
| 🙅 | Decline | Politely turn down requests |

**Management:**

Right-click the tray icon → **Personas...** to open the graphical interface:
- Create, edit, and delete personas
- Import/export JSON files (easy to share)
- Open the personas folder for direct file management

Each persona can customize:
- Insert mode prompt
- Polish mode prompt
- Independent model, temperature, and max tokens

## Development

```bash
uv run ruff check src/      # Lint
uv run ruff format src/      # Format
uv run pytest                # Run tests
```

## Roadmap

- **Distribution** — Standalone `.exe` via PyInstaller/Nuitka. No Python installation required.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
