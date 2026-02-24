# UnType (忘言)

> 得意忘言 — 说出所想，忘掉打字。

[English](README_EN.md)

**UnType** 是一个开源的 AI 语音输入工具（Windows）。它不只是转录——它会**思考**。一个快捷键，两种超能力：

1. **说话即输入** — 语音经 STT 转录后，LLM 自动润色：去除语气词（"嗯"、"那个"）、修正标点、纠正识别错误。到达光标的是润色后的成稿，不是原始语音垃圾。

2. **选中即润色** — 选中已有文字，说出修改指令（"缩短一些"、"翻译成英文"、"改成更正式的语气"），LLM 帮你改写。语音驱动的文字编辑，随处可用。

## 为什么选 UnType？

大多数语音输入工具只给你原始转录——充满"嗯"、"那个"、标点错误和识别偏差。你最终花在修正上的时间，比省下的打字时间还多。

**UnType = STT + LLM。** 语音先转录，再由 LLM 润色为干净、规范的文本——开口即终稿。需要编辑已有文字时，选中它，说话就行。

## 功能

- **AI 润色输出** — 不是原始转录。LLM 在文本到达光标前，自动修正标点、语气词、语法和识别错误。
- **语音编辑选中文字** — 选中文字，说出指令，LLM 执行修改。语音版的超级查找替换。
- **按键说话** — 按住快捷键（默认 F6）录音，松开后处理。在任何应用中都能用。
- **双语音识别后端** — 在线 API（OpenAI 兼容）或本地推理（[faster-whisper](https://github.com/SYSTRAN/faster-whisper)）。自由选择。
- **系统托盘界面** — 带颜色状态指示的托盘图标 + 设置对话框。
- **人格面具** — 录音时按数字键（1-9）或点击即可预选 LLM 人格。为不同场景定义语气配置：学术、职场、日常、要点整理——每个都有独立的提示词、模型和温度。往 `personas/` 目录放一个 JSON 文件就能新增人格。预选人格后跳过暂存区，工作流更快。

## 工作原理

```
按住快捷键 → 说话 → 松开快捷键
                ↓
   （录音期间：人格栏可见，
    按 1-9 预选人格）
                ↓
        [ STT：语音 → 原始文本 ]
                ↓
   ┌─── 已配置人格？ ───┐
   │ 是                  │ 否
   ↓                     ↓
[ LLM：使用人格 ]   [ 暂存区：编辑 ]
   ↓                     ↓
文本出现在光标处 ✓  [ LLM → 光标 ✓ ]
```

**两种模式，自动检测：**

| 模式 | 触发条件 | 效果 |
|------|---------|------|
| **插入** | 未选中文本 | 语音 → STT → LLM 润色 → 插入光标处 |
| **润色** | 已选中文本 | 语音作为指令 → LLM 修改选中的文本 |

## 快速开始

```bash
git clone https://github.com/jlmaoju/untype.git
cd untype
uv sync
uv run untype
```

1. 系统托盘出现绿色圆点。右键 → **Settings** → 填入 API 密钥。
2. 在任意输入框中点击，按住 **F6**，说话，松开 **F6**。
3. 润色好的文字出现在光标处。

## 环境要求

- Windows 10/11
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（推荐的包管理器）
- 可用的麦克风
- OpenAI 兼容的 STT API 密钥（在线模式），或 GPU（本地 Whisper 推理）
- OpenAI 兼容的 LLM API 密钥（用于文本润色；可选但推荐）

## 配置

设置存储在 `~/.untype/config.toml`（首次启动时创建）：

| 区段 | 键 | 默认值 | 说明 |
|------|-----|--------|------|
| `hotkey` | `trigger` | `f6` | 按键说话快捷键 |
| `audio` | `gain_boost` | `3.0` | 低音量语音增益倍数 |
| `stt` | `backend` | `api` | `api` 或 `local` |
| `stt` | `api_base_url` | `""` | OpenAI 兼容 STT API 端点 |
| `stt` | `api_key` | `""` | STT API 密钥 |
| `stt` | `api_model` | `gpt-4o-transcribe` | STT 模型名称 |
| `stt` | `model_size` | `small` | 本地 Whisper 模型大小 |
| `llm` | `base_url` | `""` | OpenAI 兼容 Chat API 端点 |
| `llm` | `api_key` | `""` | LLM API 密钥 |
| `llm` | `model` | `""` | LLM 模型名称 |

### 人格面具

在 `personas/` 目录放入 JSON 文件来定义人格面具。每个文件定义一个人格：

```json
{
  "id": "academic",
  "name": "学术",
  "icon": "📚",
  "prompt_polish": "",
  "prompt_insert": "You are an academic writing assistant...",
  "model": "",
  "temperature": 0.2,
  "max_tokens": null
}
```

- 文件按字母排序——用 `01_`、`02_` 前缀控制顺序。
- 所有人格在录音时显示为可点击按钮（按 1-9 预选）。
- 空字段（`""` 或 `null`）回退到全局配置。

## 开发

```bash
uv run ruff check src/      # 代码检查
uv run ruff format src/      # 代码格式化
uv run pytest                # 运行测试
```

## 开发计划

- **后悔药（Ghost Menu）** — 注入后的撤销菜单：恢复原始草稿或重新生成。没有倒计时压力——撤销选项会一直在，直到你主动关掉。
- **分发** — 通过 PyInstaller/Nuitka 打包成独立 `.exe`。无需安装 Python。

## 许可证

本项目采用 [GNU 通用公共许可证 v3.0](LICENSE) 授权。
