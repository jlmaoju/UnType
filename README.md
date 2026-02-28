# UnType (忘言)

> 筌者所以在鱼，得鱼而忘筌；蹄者所以在兔，得兔而忘蹄；言者所以在意，得意而忘言。
> — 《庄子·外物》

[English](README_EN.md)

**UnType** 是一个开源的 AI 语音输入工具（Windows）。它不只是转录——它会**思考**。一个快捷键，两种超能力：

1. **说话即输入** — 语音经 STT 转录后，LLM 自动润色：去除语气词、"嗯"、"那个"、修正标点、纠正识别错误。到达光标的是润色后的成稿。

2. **选中即润色** — 选中已有文字，说出修改指令（"缩短一些"、"翻译成英文"、"改成更正式的语气"），LLM 帮你改写。

## 为什么选 UnType？

大多数语音输入工具只给你原始转录——充满"嗯"、"那个"、标点错误和识别偏差。你最终花在修正上的时间，比省下的打字时间还多。

**UnType = STT + LLM。** 语音先转录，再由 LLM 润色为干净、规范的文本——开口即终稿。需要编辑已有文字时，选中它，说话就行。

**内置 8 种人格面具（Persona Masks）**，为不同场景预设语气：
- ✨ 默认 — 常规润色，简洁自然
- 🌙 诗意 — 隐喻修辞，华丽文风
- 👔 对领导 — 正式、得体的职场沟通
- 🤝 对同事 — 友好但专业的日常交流
- 📋 子弹点 — 自动整理成要点列表
- 🌐 英译 — 中文语音 → 英文输出
- 🗣️ 大白话 — 把复杂概念说简单
- 🙅 婉拒 — 礼貌地拒绝各种请求

录音时按数字键（1-9）即可切换，选择会自动记住。在**人格管理**中可自行增删改，也可以**停用**不需要的人格（录音时将不再显示）。

| 人格 | 场景 | 效果 |
|------|------|------|
| ✨ 默认 | 日常写作 | 简洁自然，去除口语 |
| 👔 对领导 | 职场沟通 | 正式、得体 |
| 🤝 对同事 | 团队交流 | 友好但专业 |
| 📋 子弹点 | 列表整理 | 自动分点 |
| 🌐 英译 | 中译英 | 中文语音 → 英文输出 |
| 🗣️ 大白话 | 简化表达 | 把复杂说简单 |
| 🙅 婉拒 | 拒绝请求 | 礼貌地拒绝 |
| 🌙 诗意 | 文学创作 | 华丽修辞 |

右键托盘图标 → **人格管理** 可自行增删改。

---

## 快速开始

<p align="center">
  <a href="https://github.com/jlmaoju/UnType/releases">
    <img src="https://img.shields.io/github/v/release/jlmaoju/UnType?style=for-the-badge&logo=windows&label=下载&color=0066CC" alt="Download">
  </a>
</p>

### 📥 下载预编译版本（推荐）

不想安装 Python？去 [Releases](https://github.com/jlmaoju/UnType/releases) 页面下载 `.exe` 文件，双击运行即可。

### 💻 从源码运行

```bash
git clone https://github.com/jlmaoju/UnType.git
cd untype
uv sync
uv run untype
```

### 🎯 三步上手

1. 首次运行会自动弹出**设置向导**，按提示配置 API
2. 在任意输入框中点击，按 **F6** 开始说话
3. 再按 **F6** 结束，润色好的文字自动出现

> **提示**：支持阿里云、通义千问、DeepSeek 等多种 OpenAI 兼容服务。

---

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
| `hotkey` | `mode` | `toggle` | `toggle`（按一下开始/结束）或 `hold`（按住说话） |
| `audio` | `sample_rate` | `16000` | 录音采样率（Hz） |
| `audio` | `device` | `""` | 音频设备（空字符串 = 系统默认） |
| `audio` | `gain_boost` | `1.5` | 低音量语音增益倍数 |
| `stt` | `backend` | `realtime_api` | `realtime_api`（阿里云）、`api` 或 `local` |
| `stt` | `api_base_url` | `""` | 在线 STT API 端点（OpenAI 兼容） |
| `stt` | `api_key` | `""` | 在线 STT API 密钥 |
| `stt` | `api_model` | `gpt-4o-transcribe` | 在线 STT 模型名称 |
| `stt` | `model_size` | `small` | 本地 Whisper 模型大小（`small`/`medium`/`large-v3`） |
| `stt` | `device` | `auto` | 本地推理设备（`auto`/`cuda`/`cpu`） |
| `stt` | `compute_type` | `auto` | 本地推理计算精度（`auto`/`float16`/`int8`/`int8_float16`） |
| `stt` | `beam_size` | `5` | Beam search 大小（本地模式） |
| `stt` | `vad_filter` | `true` | 是否启用 VAD 过滤（本地模式） |
| `stt` | `vad_threshold` | `0.3` | VAD 灵敏度，越低越灵敏（本地模式） |
| `stt` | `language` | `zh` | 转录语言代码 |
| `stt` | `realtime_api_key` | `""` | 阿里云实时 API 密钥 |
| `stt` | `realtime_api_model` | `paraformer-realtime-v2` | 阿里云实时模型名称 |
| `stt` | `realtime_api_format` | `pcm` | 实时 API 音频格式 |
| `stt` | `realtime_api_sample_rate` | `16000` | 实时 API 采样率 |
| `llm` | `base_url` | `""` | LLM API 端点（OpenAI 兼容） |
| `llm` | `api_key` | `""` | LLM API 密钥 |
| `llm` | `model` | `""` | LLM 模型名称 |
| `llm` | `temperature` | `0.3` | LLM 采样温度 |
| `llm` | `max_tokens` | `2048` | LLM 最大输出 token 数 |
| `language` | `language` | `zh` | 界面语言（`zh`/`en`） |
| `overlay` | `capsule_position_mode` | `"fixed"` | 胶囊位置模式：`"fixed"`（可拖动）或 `"caret"`（跟随光标） |
| `overlay` | `capsule_fixed_x` | `null` | 固定模式下的 X 位置（null = 自动居中） |
| `overlay` | `capsule_fixed_y` | `null` | 固定模式下的 Y 位置（null = 自动底部） |

### STT 后端选择

**在线 API（默认）**
- 使用 OpenAI 兼容的 `/audio/transcriptions` 接口
- 支持任意中转服务
- 录音结束后一次性返回结果

**本地模型**
- 使用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 本地推理
- 需要显卡支持（CUDA）
- 隐私性好，无需联网

**阿里云实时 API（推荐）**
- 使用阿里云 DashScope 实时语音识别
- **采用 WebSocket 流式传输，录音过程中实时显示识别文字**
- **延迟极低，体验接近微信语音输入**
- 需要申请 [阿里云 DashScope API Key](https://dashscope.console.aliyun.com/)

## 故障排除

### 快捷键无响应
- 检查是否有其他应用占用了相同的快捷键
- 尝试在设置中更改为其他快捷键（如 `F7`、`Ctrl+Space`）
- 以管理员身份运行程序

### 文字注入到错误位置
- 确保在录音开始时目标窗口是活动窗口
- 如果切换了窗口，结果会被暂存到气泡中，切回原窗口后点击气泡即可注入

### STT 识别不准确
- 尝试增加 `audio.gain_boost` 值（默认 1.5，可调至 3.0 或更高）
- 检查麦克风设置，确保输入设备正确
- 尝试更换 STT 后端（在线 API 通常比本地模型更准确）

### LLM 润色效果不佳
- 尝试调整 `llm.temperature`（越低越保守，越高越 creative）
- 检查 LLM API 配置是否正确
- 尝试使用不同的人格面具

### 托盘图标状态异常
- **绿色**：正常运行
- **黄色**：API 配置缺失或不完整
- **红色**：API 调用失败或网络错误

## 常见问题

### UnType 是免费的吗？
是的，UnType 是完全开源免费的。但你需要自行提供 STT 和 LLM API 密钥（相关费用由 API 提供商收取）。

### 支持 macOS 吗？
目前仅支持 Windows。macOS 支持计划在未来版本中添加。

### 本地模式需要什么配置？
- 需要支持 CUDA 的 NVIDIA 显卡
- 推荐显存 4GB 以上
- 需要安装 CUDA Toolkit

### 如何添加自定义人格面具？
1. 右键托盘图标 → **人格管理**
2. 点击 **+ 新建** 按钮创建新人格
3. 填写名称、图标、提示词等字段
4. 勾选 **激活** 选项使其在录音时显示
5. 或者直接在 `personas/` 目录中创建 `.json` 文件

### 录音时可以切换人格吗？
可以。在录音过程中按数字键 1-9 即可预选人格，录音结束后会直接使用预选的人格进行润色。只有**已激活**的人格才会显示在录音时的人格栏中。

## 工作原理

```
按一下快捷键 → 说话 → 再按一下结束
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
                ↓
       （后悔药菜单出现）
```

**两种模式，自动检测：**

| 模式 | 触发条件 | 效果 |
|------|---------|------|
| **插入** | 未选中文本 | 语音 → STT → LLM 润色 → 插入光标处 |
| **润色** | 已选中文本 | 语音作为指令 → LLM 修改选中的文本 |

## 开发

```bash
uv run ruff check src/      # 代码检查
uv run ruff format src/      # 代码格式化
uv run pytest                # 运行测试
```

## 许可证

本项目采用 [GNU 通用公共许可证 v3.0](LICENSE) 授权。

## 更新日志

### v0.3.0 (2025-02-28)
- **新增首次运行向导**：新用户首次启动时自动弹出配置向导，引导完成 STT 和 LLM 配置
- **新增人格激活功能**：可在人格管理中启用/禁用人格，录音时只显示已激活的人格
- **新增 LLM 连接验证**：向导和设置中可测试 LLM API 连接是否正常
- **改进配置体验**：向导支持实时预览配置、验证 API 连接、选择要激活的人格
- **优化界面语言**：更新中英文翻译，新增人格激活相关文本

### v0.2.1 (2025-02-26)
- 新增"默认"人格面具（常规润色风格）
- 新增"诗意"人格面具（华丽修辞文风）
- 新增人格记忆功能，自动记住上次选择的人格
- 新增录音时长显示（胶囊上显示如 "1:23"）
- 新增录音超时保护（5 分钟自动停止，防止过度消耗）
- 新增日志记录功能（设置中可打开日志文件夹）
- 默认 STT 后端改为阿里云实时 API
- 默认音频增益调整为 1.5
- "打开日志"功能移至设置对话框

### v0.2.0 (2025-02-25)
- 新增阿里云实时语音识别后端，录音过程中实时显示识别文字
- 新增胶囊位置固定模式（可拖动，位置会被记住）
- 新增设置界面字段动态显示（根据后端选择显示/隐藏相关配置）
- 修复快捷键切换时的竞态条件问题
- 修复快捷键黑名单（防止与系统快捷键冲突）
- 修复后悔菜单位置跟随胶囊配置
