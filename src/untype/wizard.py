"""First-run setup wizard for UnType."""

import logging
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

import httpx

from untype.config import AppConfig, save_config
from untype.stt import STTEngine, STTApiEngine, STTRealtimeApiEngine

logger = logging.getLogger(__name__)


# First run marker file
FIRST_RUN_MARKER = os.path.expanduser("~/.untype/.first_run_completed")
WIZARD_STATE_FILE = os.path.expanduser("~/.untype/.wizard_state.json")


def is_first_run() -> bool:
    """Check if this is the first run."""
    return not os.path.exists(FIRST_RUN_MARKER)


def mark_first_run_complete() -> None:
    """Mark the first run as complete."""
    try:
        os.makedirs(os.path.dirname(FIRST_RUN_MARKER), exist_ok=True)
        with open(FIRST_RUN_MARKER, "w") as f:
            f.write("completed")
    except Exception as e:
        logger.warning("Failed to mark first run complete: %s", e)


class SetupWizard:
    """First-run setup wizard for UnType configuration."""

    def __init__(
        self,
        config: AppConfig,
        on_complete: Callable[[AppConfig], None],
    ):
        self._config = config
        self._on_complete = on_complete

        # Wizard state
        self._current_page = 0
        self._pages: list = []
        self._page_vars: dict = {}

        # Temporary config (not saved until completion)
        self._temp_config: AppConfig = None

        # UI
        self._root: tk.Tk = None
        self._content_frame: tk.Frame = None

    def _create_string_var(self, value: str = "") -> tk.StringVar:
        """Create a StringVar bound to this wizard's root."""
        if self._root is None:
            # Fallback if called before root is created
            return tk.StringVar(value=value)
        return tk.StringVar(self._root, value=value)

    def _create_boolean_var(self, value: bool = False) -> tk.BooleanVar:
        """Create a BooleanVar bound to this wizard's root."""
        if self._root is None:
            return tk.BooleanVar(value=value)
        return tk.BooleanVar(self._root, value=value)

    def run(self) -> None:
        """Run the wizard."""
        self._create_ui()
        self._init_pages()
        self._show_page(0)
        self._root.mainloop()

    def _create_ui(self) -> None:
        """Create the wizard window."""
        self._root = tk.Tk()
        self._root.title("UnType 设置向导")
        self._root.geometry("680x650")
        self._root.resizable(False, False)

        # Center window
        self._root.update_idletasks()
        w = self._root.winfo_width()
        h = self._root.winfo_height()
        x = (self._root.winfo_screenwidth() - w) // 2
        y = (self._root.winfo_screenheight() - h) // 2
        self._root.geometry(f"+{x}+{y}")

        # Main frame with padding
        main_frame = tk.Frame(self._root, bg="#f5f5f5", padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        # Header frame
        header_frame = tk.Frame(main_frame, bg="#f5f5f5", height=30)
        header_frame.pack(fill="x", pady=(0, 15))
        header_frame.pack_propagate(False)

        # Content frame (scrollable area)
        self._content_frame = tk.Frame(main_frame, bg="white", relief="solid", borderwidth=1)
        self._content_frame.pack(fill="both", expand=True, pady=(0, 15))

        # Navigation frame with fixed height
        nav_frame = tk.Frame(main_frame, bg="#f5f5f5", height=50)
        nav_frame.pack(fill="x", side="bottom")
        nav_frame.pack_propagate(False)

        # Back button
        self._back_btn = tk.Button(
            nav_frame,
            text="< 上一步",
            font=("Microsoft YaHei UI", 10),
            bg="#e0e0e0",
            fg="#333333",
            activebackground="#d0d0d0",
            activeforeground="#333333",
            relief="flat",
            width=10,
            pady=8,
            command=self._on_back,
        )
        self._back_btn.pack(side="left")

        # Next button
        self._next_btn = tk.Button(
            nav_frame,
            text="下一步 >",
            font=("Microsoft YaHei UI", 10),
            bg="#4CAF50",
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            relief="flat",
            width=10,
            pady=8,
            command=self._on_next,
        )
        self._next_btn.pack(side="right")

    def _on_close(self) -> None:
        """Handle close button click."""
        if messagebox.askyesno(
            "退出设置向导",
            "设置尚未完成，确定要退出吗？\n\n退出后可以稍后在设置中重新配置。"
        ):
            self._root.destroy()

    def _on_back(self) -> None:
        """Handle back button click."""
        if self._current_page > 0:
            self._show_page(self._current_page - 1)

    def _on_next(self) -> None:
        """Handle next button click."""
        # Validate current page before moving
        if not self._validate_current_page():
            return

        if self._current_page < len(self._pages) - 1:
            self._show_page(self._current_page + 1)
        else:
            # Complete the wizard
            self._complete_wizard()

    def _show_page(self, page_num: int) -> None:
        """Show a specific page."""
        self._current_page = page_num

        # Clear content frame
        for widget in self._content_frame.winfo_children():
            widget.destroy()

        # Show current page
        self._pages[page_num](self._content_frame)

        # Update navigation buttons
        self._back_btn.config(state="normal" if page_num > 0 else "disabled")

        is_last = page_num == len(self._pages) - 1
        self._next_btn.config(text="完成" if is_last else "下一步 >")

    def _validate_current_page(self) -> bool:
        """Validate the current page before moving to next."""
        page_validators = {
            0: lambda: True,  # Welcome page
            1: self._validate_stt_selection,
            2: self._validate_stt_config,
            3: self._validate_stt_verify,
            4: self._validate_llm_config,
            5: self._validate_llm_verify,
            6: self._validate_persona_selection,
            7: self._validate_quick_start,
        }
        validator = page_validators.get(self._current_page, lambda: True)
        return validator()

    def _complete_wizard(self) -> None:
        """Complete the wizard and save configuration."""
        try:
            # Apply temp config to actual config
            if self._temp_config:
                # Copy STT settings
                self._config.stt.backend = self._temp_config.stt.backend
                self._config.stt.api_base_url = self._temp_config.stt.api_base_url
                self._config.stt.api_key = self._temp_config.stt.api_key
                self._config.stt.api_model = self._temp_config.stt.api_model
                self._config.stt.realtime_api_key = self._temp_config.stt.realtime_api_key
                self._config.stt.realtime_api_model = self._temp_config.stt.realtime_api_model
                self._config.stt.model_size = self._temp_config.stt.model_size

                # Copy LLM settings
                self._config.llm.base_url = self._temp_config.llm.base_url
                self._config.llm.api_key = self._temp_config.llm.api_key
                self._config.llm.model = self._temp_config.llm.model

                # Save config
                save_config(self._config)

            # Mark first run as complete
            mark_first_run_complete()

            # Clean up tkinter variables to avoid thread conflicts
            self._cleanup_vars()

            # Close wizard - use quit() to stop mainloop first
            self._root.quit()
            self._root.update_idletasks()  # Process pending events
            self._root.destroy()

            # Notify completion (after tkinter is fully cleaned up)
            self._on_complete(self._config)

        except Exception as e:
            messagebox.showerror("保存失败", f"保存配置时出错：{e}")
            logger.exception("Failed to save wizard configuration")

    def _cleanup_vars(self) -> None:
        """Clean up tkinter variables to prevent thread conflicts."""
        # Clear all StringVar and BooleanVar references
        for key, var in list(self._page_vars.items()):
            try:
                if hasattr(var, 'set'):
                    var.set("")  # Clear value
                del var
            except Exception:
                pass
        self._page_vars.clear()

    # ------------------------------------------------------------------
    # Page implementations
    # ------------------------------------------------------------------

    def _page_welcome(self, parent: tk.Frame) -> None:
        """Show welcome page (Page 0)."""
        frame = tk.Frame(parent, bg="#f8f9fa", padx=40, pady=40)
        frame.pack(fill="both", expand=True)

        # Title
        tk.Label(
            frame,
            text="🎯 欢迎使用 UnType (忘言)",
            font=("Microsoft YaHei UI", 18, "bold"),
            bg="#f8f9fa",
            fg="#333333",
        ).pack(pady=(0, 8))

        tk.Label(
            frame,
            text="AI 语音输入 + 文本润色，两种模式，随心切换",
            font=("Microsoft YaHei UI", 11),
            bg="#f8f9fa",
            fg="#666666",
        ).pack(pady=(0, 30))

        # Two-column layout for modes
        modes_frame = tk.Frame(frame, bg="#f8f9fa")
        modes_frame.pack(fill="both", expand=True, pady=20)

        # Left card - Speak to Insert
        left_card = tk.Frame(modes_frame, bg="white", relief="solid", borderwidth=1, padx=25, pady=20)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Mode icon and title
        tk.Label(
            left_card,
            text="🎤 说话即输入",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#2196F3",
        ).pack(pady=(0, 12))

        # Flow description
        flow_frame = tk.Frame(left_card, bg="white")
        flow_frame.pack(pady=(0, 15))

        flow_steps = [
            ("1. 按下 F6", "#555"),
            ("↓", "#999"),
            ("2. 说话", "#555"),
            ("↓", "#999"),
            ("3. 松开 → 润色后的文本出现在光标处", "#2196F3"),
        ]

        for step, color in flow_steps:
            tk.Label(
                flow_frame,
                text=step,
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg=color,
            ).pack(anchor="center", pady=2)

        # Divider
        tk.Frame(left_card, bg="#e0e0e0", height=1).pack(fill="x", pady=12)

        # Use cases
        tk.Label(
            left_card,
            text="💬 适用场景",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="white",
            fg="#555",
        ).pack(anchor="w", pady=(0, 8))

        use_cases = ["• 快速写作 / 记笔记", "• 回复消息", "• 撰写邮件"]
        for case in use_cases:
            tk.Label(
                left_card,
                text=case,
                font=("Microsoft YaHei UI", 9),
                bg="white",
                fg="#666",
            ).pack(anchor="w", pady=3)

        # Right card - Select to Polish
        right_card = tk.Frame(modes_frame, bg="white", relief="solid", borderwidth=1, padx=25, pady=20)
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

        # Mode icon and title
        tk.Label(
            right_card,
            text="✏️ 选中即润色",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#FF9800",
        ).pack(pady=(0, 12))

        # Flow description
        flow_frame2 = tk.Frame(right_card, bg="white")
        flow_frame2.pack(pady=(0, 15))

        flow_steps2 = [
            ("1. 选中已有文字", "#555"),
            ("↓", "#999"),
            ("2. 说出修改要求", "#555"),
            ("↓", "#999"),
            ("3. AI 按你的要求改写", "#FF9800"),
        ]

        for step, color in flow_steps2:
            tk.Label(
                flow_frame2,
                text=step,
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg=color,
            ).pack(anchor="center", pady=2)

        # Divider
        tk.Frame(right_card, bg="#e0e0e0", height=1).pack(fill="x", pady=12)

        # Example commands
        tk.Label(
            right_card,
            text="💡 说话示例",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="white",
            fg="#555",
        ).pack(anchor="w", pady=(0, 8))

        examples = [
            '"把这段改得更正式"',
            '"翻译成英文"',
            '"缩短一些"',
            '"用更随意的语气"',
        ]
        for example in examples:
            tk.Label(
                right_card,
                text=example,
                font=("Microsoft YaHei UI", 9),
                bg="white",
                fg="#666",
            ).pack(anchor="w", pady=3)

        # Bottom hint
        tk.Label(
            frame,
            text='👇 点击"下一步"开始配置你的 API',
            font=("Microsoft YaHei UI", 10),
            bg="#f8f9fa",
            fg="#999",
        ).pack(pady=(20, 0))

    def _page_stt_selection(self, parent: tk.Frame) -> None:
        """Show STT mode selection page (Page 1)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        # Title
        tk.Label(
            frame,
            text="选择语音识别方式",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 20))

        tk.Label(
            frame,
            text="请选择您偏好的语音识别模式：",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#666666",
        ).pack(pady=(0, 20))

        # Initialize temp config if needed
        if self._temp_config is None:
            import copy
            self._temp_config = copy.deepcopy(self._config)

        # Selection variable
        if "stt_backend" not in self._page_vars:
            # Default to realtime_api
            self._page_vars["stt_backend"] = self._create_string_var(
                value=self._temp_config.stt.backend or "realtime_api"
            )

        # Options
        options = [
            {
                "value": "realtime_api",
                "title": "🌐 阿里云实时 API",
                "subtitle": "低延迟、实时预览、需联网",
                "badge": "推荐",
            },
            {
                "value": "api",
                "title": "🔗 在线 API",
                "subtitle": "兼容 OpenAI 格式的 API",
                "badge": None,
            },
            {
                "value": "local",
                "title": "💾 本地模型",
                "subtitle": "无需联网、首次需下载模型",
                "badge": None,
            },
        ]

        for opt in options:
            card_frame = tk.Frame(
                frame,
                bg="white",
                relief="solid",
                borderwidth=1,
                padx=15,
                pady=12,
            )
            card_frame.pack(fill="x", pady=8)

            # Radio button on left
            radio = tk.Radiobutton(
                card_frame,
                text="",
                variable=self._page_vars["stt_backend"],
                value=opt["value"],
                bg="white",
                activebackground="white",
                font=("Arial", 14),
            )
            radio.pack(side="left", padx=(0, 10))

            # Content frame
            content_frame = tk.Frame(card_frame, bg="white")
            content_frame.pack(side="left", fill="both", expand=True)

            # Title row
            title_row = tk.Frame(content_frame, bg="white")
            title_row.pack(fill="x")

            tk.Label(
                title_row,
                text=opt["title"],
                font=("Microsoft YaHei UI", 11, "bold"),
                bg="white",
                fg="#333333",
            ).pack(side="left")

            if opt["badge"]:
                badge = tk.Label(
                    title_row,
                    text=f" {opt['badge']} ",
                    font=("Microsoft YaHei UI", 8),
                    bg="#4CAF50",
                    fg="white",
                )
                badge.pack(side="left", padx=(8, 0))

            # Subtitle
            tk.Label(
                content_frame,
                text=opt["subtitle"],
                font=("Microsoft YaHei UI", 9),
                bg="white",
                fg="#888888",
            ).pack(anchor="w")

            # Click to select
            for widget in [card_frame, content_frame, title_row]:
                widget.bind("<Button-1>", lambda e, v=opt["value"]: self._page_vars["stt_backend"].set(v))

    def _page_stt_config_realtime(self, parent: tk.Frame) -> None:
        """Show realtime API config page (Page 2 - realtime)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        # Title
        tk.Label(
            frame,
            text="配置阿里云 API",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 10))

        tk.Label(
            frame,
            text="请输入您的阿里云 DashScope API 密钥：",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#666666",
        ).pack(pady=(0, 20))

        # API Key input
        if "realtime_api_key" not in self._page_vars:
            self._page_vars["realtime_api_key"] = self._create_string_var(value=self._temp_config.stt.realtime_api_key or "")
        if "api_verified" not in self._page_vars:
            self._page_vars["api_verified"] = self._create_boolean_var(value=False)

        input_frame = tk.Frame(frame, bg="white")
        input_frame.pack(fill="x", pady=(0, 20))

        tk.Label(
            input_frame,
            text="API 密钥",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#555555",
        ).pack(anchor="w")

        entry = tk.Entry(
            input_frame,
            textvariable=self._page_vars["realtime_api_key"],
            font=("Microsoft YaHei UI", 10),
            show="*",
        )
        entry.pack(fill="x", pady=(5, 0))
        entry.bind("<FocusOut>", lambda e: self._verify_api_key())

        # Help text
        help_frame = tk.Frame(frame, bg="#f0f8ff", padx=15, pady=12)
        help_frame.pack(fill="x", pady=(0, 20))

        tk.Label(
            help_frame,
            text="💡 如何获取 API 密钥？",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="#f0f8ff",
            fg="#333333",
        ).pack(anchor="w")

        steps = [
            "1. 访问阿里云 DashScope 控制台",
            "2. 创建 API-KEY",
            "3. 复制密钥粘贴到上方",
        ]
        for step in steps:
            tk.Label(
                help_frame,
                text=step,
                font=("Microsoft YaHei UI", 9),
                bg="#f0f8ff",
                fg="#666666",
            ).pack(anchor="w", pady=2)

        tk.Button(
            help_frame,
            text="打开控制台",
            font=("Microsoft YaHei UI", 9),
            bg="#2196F3",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=lambda: os.startfile("https://dashscope.console.aliyun.com/apiKey"),
        ).pack(anchor="w", pady=(8, 0))

        # Verify status
        self._verify_status_frame = tk.Frame(frame, bg="white")
        self._verify_status_frame.pack(fill="x")

        self._verify_label = tk.Label(
            self._verify_status_frame,
            text="点击「下一步」时将验证密钥",
            font=("Microsoft YaHei UI", 9),
            bg="white",
            fg="#888888",
        )
        self._verify_label.pack()

    def _page_stt_config_api(self, parent: tk.Frame) -> None:
        """Show online API config page (Page 2 - api)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="配置在线 API",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 20))

        # Initialize variables
        if "api_base_url" not in self._page_vars:
            self._page_vars["api_base_url"] = self._create_string_var(value=self._temp_config.stt.api_base_url or "")
        if "api_key" not in self._page_vars:
            self._page_vars["api_key"] = self._create_string_var(value=self._temp_config.stt.api_key or "")
        if "api_model" not in self._page_vars:
            self._page_vars["api_model"] = self._create_string_var(value=self._temp_config.stt.api_model or "gpt-4o-transcribe")

        # Fields
        fields = [
            ("API 地址", "api_base_url", "https://api.openai.com/v1"),
            ("API 密钥", "api_key", ""),
            ("模型名称", "api_model", "gpt-4o-transcribe"),
        ]

        for label, var_key, placeholder in fields:
            field_frame = tk.Frame(frame, bg="white")
            field_frame.pack(fill="x", pady=(0, 15))

            tk.Label(
                field_frame,
                text=label,
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg="#555555",
            ).pack(anchor="w")

            tk.Entry(
                field_frame,
                textvariable=self._page_vars[var_key],
                font=("Microsoft YaHei UI", 10),
                show="*" if "key" in var_key else None,
            ).pack(fill="x", pady=(5, 0))

    def _page_stt_config_local(self, parent: tk.Frame) -> None:
        """Show local model config page (Page 2 - local)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="下载本地模型",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 10))

        tk.Label(
            frame,
            text="本地模式需要下载 Whisper 模型：",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#666666",
        ).pack(pady=(0, 20))

        # Model selection
        if "local_model_size" not in self._page_vars:
            self._page_vars["local_model_size"] = self._create_string_var(value="small")
        if "model_downloading" not in self._page_vars:
            self._page_vars["model_downloading"] = self._create_boolean_var(value=False)
        if "model_downloaded" not in self._page_vars:
            self._page_vars["model_downloaded"] = self._create_boolean_var(value=self._check_local_model_exists("small"))

        # Info card
        info_frame = tk.Frame(frame, bg="#f0f0f0", padx=15, pady=12)
        info_frame.pack(fill="x", pady=(0, 20))

        tk.Label(
            info_frame,
            text="模型大小：small (推荐)",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="#f0f0f0",
            fg="#333333",
        ).pack(anchor="w")

        tk.Label(
            info_frame,
            text="• 下载大小：约 500MB  • 适合日常使用  • CPU 运行流畅",
            font=("Microsoft YaHei UI", 9),
            bg="#f0f0f0",
            fg="#666666",
        ).pack(anchor="w", pady=(5, 0))

        # Model size selection
        tk.Label(
            frame,
            text="选择模型大小：",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#555555",
        ).pack(anchor="w", pady=(0, 10))

        models = [
            ("small", "推荐，约 500MB"),
            ("medium", "约 1.5GB"),
            ("large-v3", "约 3GB"),
        ]

        model_frame = tk.Frame(frame, bg="white")
        model_frame.pack(fill="x")

        for value, desc in models:
            row = tk.Frame(model_frame, bg="white")
            row.pack(fill="x", pady=3)
            tk.Radiobutton(
                row,
                text=f"{value:12} - {desc}",
                variable=self._page_vars["local_model_size"],
                value=value,
                bg="white",
                font=("Microsoft YaHei UI", 9),
            ).pack(anchor="w")

        # Download button / progress
        self._download_frame = tk.Frame(frame, bg="white")
        self._download_frame.pack(fill="x", pady=(20, 0))

        if self._page_vars["model_downloaded"].get():
            tk.Label(
                self._download_frame,
                text="✓ 模型已下载",
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg="#4CAF50",
            ).pack()
        else:
            self._download_btn = tk.Button(
                self._download_frame,
                text="开始下载",
                font=("Microsoft YaHei UI", 10),
                bg="#4CAF50",
                fg="white",
                relief="flat",
                cursor="hand2",
                command=self._start_model_download,
            )
            self._download_btn.pack()

        # Progress frame (hidden initially)
        self._progress_frame = tk.Frame(frame, bg="white")
        self._progress_label = tk.Label(
            self._progress_frame,
            text="正在下载...",
            font=("Microsoft YaHei UI", 9),
            bg="white",
            fg="#666666",
        )
        self._progress_bar = ttk.Progressbar(
            self._progress_frame,
            mode="indeterminate",
            length=300,
        )

    def _get_page_2(self) -> callable:
        """Get the appropriate Page 2 based on STT selection."""
        # Initialize temp config if needed
        if self._temp_config is None:
            import copy
            self._temp_config = copy.deepcopy(self._config)

        backend_var = self._page_vars.get("stt_backend")
        if backend_var is None:
            # Initialize with default from config
            backend_var = self._create_string_var(value=self._temp_config.stt.backend or "realtime_api")
            self._page_vars["stt_backend"] = backend_var
        backend = backend_var.get()
        if backend == "api":
            return self._page_stt_config_api
        elif backend == "local":
            return self._page_stt_config_local
        else:
            return self._page_stt_config_realtime

    def _page_stt_verify(self, parent: tk.Frame) -> None:
        """Show STT verification page (Page 3)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=40)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="配置验证",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 20))

        # Verification will happen when page is shown
        self._verify_frame = tk.Frame(frame, bg="white")
        self._verify_frame.pack(fill="both", expand=True)

        # Trigger verification after UI is ready
        self._root.after(100, self._verify_stt_config)

    def _page_llm_config(self, parent: tk.Frame) -> None:
        """Show LLM config page (Page 4)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="配置文本润色功能",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 10))

        tk.Label(
            frame,
            text="UnType 使用 AI 在转录后润色文本，这是核心功能。",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#666666",
        ).pack(pady=(0, 20))

        # Initialize variables
        if "llm_base_url" not in self._page_vars:
            self._page_vars["llm_base_url"] = self._create_string_var(value=self._temp_config.llm.base_url or "")
        if "llm_api_key" not in self._page_vars:
            self._page_vars["llm_api_key"] = self._create_string_var(value=self._temp_config.llm.api_key or "")
        if "llm_model" not in self._page_vars:
            self._page_vars["llm_model"] = self._create_string_var(value=self._temp_config.llm.model or "")

        # Fields
        fields = [
            ("API 地址", "llm_base_url", ""),
            ("API 密钥", "llm_api_key", ""),
            ("模型名称", "llm_model", ""),
        ]

        for label, var_key, placeholder in fields:
            field_frame = tk.Frame(frame, bg="white")
            field_frame.pack(fill="x", pady=(0, 15))

            tk.Label(
                field_frame,
                text=label,
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg="#555555",
            ).pack(anchor="w")

            tk.Entry(
                field_frame,
                textvariable=self._page_vars[var_key],
                font=("Microsoft YaHei UI", 10),
                show="*" if "key" in var_key else None,
            ).pack(fill="x", pady=(5, 0))

        # Help text
        help_frame = tk.Frame(frame, bg="#fff8e1", padx=15, pady=12)
        help_frame.pack(fill="x", pady=(10, 0))

        tk.Label(
            help_frame,
            text="💡 使用任何兼容 OpenAI API 格式的服务",
            font=("Microsoft YaHei UI", 9),
            bg="#fff8e1",
            fg="#666666",
        ).pack(anchor="w")

    def _page_llm_verify(self, parent: tk.Frame) -> None:
        """Show LLM verification page (Page 5)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=40)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="配置验证",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 20))

        # Verification will happen when page is shown
        self._llm_verify_frame = tk.Frame(frame, bg="white")
        self._llm_verify_frame.pack(fill="both", expand=True)

        # Trigger verification after UI is ready
        self._root.after(100, self._verify_llm_config)

    def _page_quick_start(self, parent: tk.Frame) -> None:
        """Show quick start page (Page 5)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="🎉 设置完成！",
            font=("Microsoft YaHei UI", 16, "bold"),
            bg="white",
            fg="#4CAF50",
        ).pack(pady=(0, 10))

        tk.Label(
            frame,
            text="现在可以开始使用 UnType 了",
            font=("Microsoft YaHei UI", 11),
            bg="white",
            fg="#666666",
        ).pack(pady=(0, 25))

        # How to use
        howto_frame = tk.Frame(frame, bg="#f9f9f9", padx=20, pady=15)
        howto_frame.pack(fill="x", pady=(0, 20))

        tk.Label(
            howto_frame,
            text="如何使用：",
            font=("Microsoft YaHei UI", 11, "bold"),
            bg="#f9f9f9",
            fg="#333333",
        ).pack(anchor="w")

        steps = [
            "1. 将光标放在想要输入的位置",
            "2. 按下 F6 键开始说话",
            "3. 再次按下 F6 停止",
            "4. 文本自动插入到光标位置",
        ]

        for step in steps:
            tk.Label(
                howto_frame,
                text=step,
                font=("Microsoft YaHei UI", 9),
                bg="#f9f9f9",
                fg="#555555",
            ).pack(anchor="w", pady=3)

        # Tips
        tips_frame = tk.Frame(frame, bg="white")
        tips_frame.pack(fill="x", pady=(0, 20))

        tk.Label(
            tips_frame,
            text="💡 小技巧：",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="white",
            fg="#555555",
        ).pack(anchor="w")

        tips = [
            "说话时可以看到实时预览",
            "支持选择文本后说话进行润色",
            "可在设置中自定义热键",
        ]

        for tip in tips:
            tk.Label(
                tips_frame,
                text=f"• {tip}",
                font=("Microsoft YaHei UI", 9),
                bg="white",
                fg="#888888",
            ).pack(anchor="w", pady=2)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    def _validate_stt_selection(self) -> bool:
        """Validate STT selection page."""
        backend = self._page_vars["stt_backend"].get()
        if not backend:
            messagebox.showwarning("请选择", "请选择一种语音识别方式")
            return False
        self._temp_config.stt.backend = backend
        return True

    def _validate_stt_config(self) -> bool:
        """Validate STT config page."""
        backend = self._temp_config.stt.backend

        if backend == "realtime_api":
            api_key = self._page_vars["realtime_api_key"].get().strip()
            if not api_key:
                messagebox.showwarning("请输入密钥", "请输入阿里云 API 密钥")
                return False
            if not api_key.startswith("sk-"):
                messagebox.showwarning("格式错误", "API 密钥应该以 'sk-' 开头")
                return False
            self._temp_config.stt.realtime_api_key = api_key
            return True

        elif backend == "api":
            base_url = self._page_vars["api_base_url"].get().strip()
            api_key = self._page_vars["api_key"].get().strip()
            model = self._page_vars["api_model"].get().strip()

            if not base_url or not api_key or not model:
                messagebox.showwarning("请填写完整", "请填写所有 API 配置信息")
                return False

            self._temp_config.stt.api_base_url = base_url
            self._temp_config.stt.api_key = api_key
            self._temp_config.stt.api_model = model
            return True

        elif backend == "local":
            if not self._page_vars["model_downloaded"].get():
                messagebox.showwarning("请下载模型", "请先下载本地模型")
                return False
            self._temp_config.stt.model_size = self._page_vars["local_model_size"].get()
            return True

        return True

    def _validate_stt_verify(self) -> bool:
        """Validate STT verification page."""
        # Verification happens on page show, this just checks if it passed
        return self._page_vars.get("stt_verified", False)

    def _validate_llm_config(self) -> bool:
        """Validate LLM config page."""
        base_url = self._page_vars["llm_base_url"].get().strip()
        api_key = self._page_vars["llm_api_key"].get().strip()
        model = self._page_vars["llm_model"].get().strip()

        if not base_url or not api_key or not model:
            messagebox.showwarning("请填写完整", "请填写所有 LLM 配置信息")
            return False

        self._temp_config.llm.base_url = base_url
        self._temp_config.llm.api_key = api_key
        self._temp_config.llm.model = model
        return True

    def _validate_llm_verify(self) -> bool:
        """Validate LLM verification page."""
        # Verification happens on page show, this just checks if it passed
        return self._page_vars.get("llm_verified", False)

    def _validate_quick_start(self) -> bool:
        """Validate quick start page (always true)."""
        return True

    def _validate_persona_selection(self) -> bool:
        """Validate and save persona activation states."""
        from untype.config import get_personas_dir, Persona, save_persona
        import json

        # Get the checkbox states
        checkboxes = self._page_vars.get("persona_checkboxes", {})
        if not checkboxes:
            return True  # No personas to configure

        personas_dir = get_personas_dir()
        if not personas_dir.is_dir():
            return True

        # Update each persona file with the active state
        for persona_id, checkbox_var in checkboxes.items():
            is_active = checkbox_var.get()

            # Read the persona file
            path = personas_dir / f"{persona_id}.json"
            if not path.exists():
                continue

            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)

                # Update the active field
                data["active"] = is_active

                # Save the updated persona
                persona = Persona(**data)
                save_persona(persona)

            except Exception as e:
                logger.warning("Failed to update persona %s: %s", persona_id, e)

        return True

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _verify_api_key(self) -> bool:
        """Verify the DashScope API key."""
        if hasattr(self, '_verifying') and self._verifying:
            return False

        api_key_var = self._page_vars.get("realtime_api_key")
        if api_key_var is None:
            return False
        api_key = api_key_var.get().strip()
        if not api_key or not api_key.startswith("sk-"):
            return False

        self._verifying = True
        self._verify_label.config(text="正在验证密钥...", fg="#2196F3")

        def verify_thread():
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(
                        "https://dashscope.aliyuncs.com/api/v1/services",
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                # Any response means service is reachable
                self._root.after(0, lambda: self._verify_label.config(text="✓ 密钥验证通过", fg="#4CAF50"))
                self._page_vars["api_verified"].set(True)
            except Exception:
                self._root.after(0, lambda: self._verify_label.config(text="✗ 密钥验证失败", fg="#f44336"))
                self._page_vars["api_verified"].set(False)
            finally:
                self._verifying = False

        threading.Thread(target=verify_thread, daemon=True).start()
        return False  # Don't auto-advance

    def _verify_stt_config(self) -> None:
        """Verify STT configuration."""
        # Clear frame and show loading
        for widget in self._verify_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self._verify_frame,
            text="正在验证配置...",
            font=("Microsoft YaHei UI", 11),
            bg="white",
            fg="#2196F3",
        ).pack(pady=30)

        self._root.update()

        backend = self._temp_config.stt.backend
        verified = False
        message = ""
        can_continue = False  # Whether user can proceed despite verification failure

        try:
            if backend == "realtime_api":
                api_key = self._temp_config.stt.realtime_api_key
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(
                        "https://dashscope.aliyuncs.com/api/v1/services",
                        headers={"Authorization": f"Bearer {api_key}"}
                    )
                verified = True
                message = "✓ 语音识别配置正常\n✓ API 密钥验证通过"
                can_continue = True

            elif backend == "api":
                verified = True
                message = "✓ API 配置已保存\n（将在首次使用时验证）"
                can_continue = True

            elif backend == "local":
                model_size = self._temp_config.stt.model_size
                if self._check_local_model_exists(model_size):
                    verified = True
                    message = "✓ 本地模型已下载"
                    can_continue = True
                else:
                    message = "✗ 本地模型不存在\n请返回上一步下载模型"
                    can_continue = False

        except Exception as e:
            message = f"✗ 验证失败：{e}\n\n可能是网络问题，您可以稍后重试，或点击「继续」稍后再验证。"
            can_continue = True  # Allow user to proceed despite network error

        # Update UI
        for widget in self._verify_frame.winfo_children():
            widget.destroy()

        if verified:
            tk.Label(
                self._verify_frame,
                text=message,
                font=("Microsoft YaHei UI", 11),
                bg="white",
                fg="#4CAF50",
                justify="left",
            ).pack(pady=20)

            tk.Label(
                self._verify_frame,
                text="配置完成！可以继续下一步了。",
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg="#666666",
            ).pack()

            self._page_vars["stt_verified"] = True
        else:
            tk.Label(
                self._verify_frame,
                text=message,
                font=("Microsoft YaHei UI", 11),
                bg="white",
                fg="#f44336" if not can_continue else "#FF9800",
                justify="left",
            ).pack(pady=20)

            btn_frame = tk.Frame(self._verify_frame, bg="white")
            btn_frame.pack(pady=10)

            tk.Button(
                btn_frame,
                text="重试",
                font=("Microsoft YaHei UI", 10),
                bg="#2196F3",
                fg="white",
                relief="flat",
                cursor="hand2",
                command=self._verify_stt_config,
            ).pack(side="left", padx=5)

            if can_continue:
                tk.Button(
                    btn_frame,
                    text="继续",
                    font=("Microsoft YaHei UI", 10),
                    bg="#4CAF50",
                    fg="white",
                    relief="flat",
                    cursor="hand2",
                    command=lambda: self._page_vars.update({"stt_verified": True}) or self._on_next(),
                ).pack(side="left", padx=5)

            self._page_vars["stt_verified"] = False

    def _check_local_model_exists(self, model_size: str) -> bool:
        """Check if local model exists."""
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if not os.path.exists(cache_dir):
                return False

            model_name = f"Systran/faster-whisper-{model_size}"
            for item in os.listdir(cache_dir):
                if model_name.replace("-", "_").replace("/", "--") in item or model_name in item:
                    return True
            return False
        except Exception:
            return False

    def _start_model_download(self) -> None:
        """Start downloading the local model."""
        if self._page_vars["model_downloading"].get():
            return

        model_size = self._page_vars["local_model_size"].get()

        # Hide download button, show progress
        if hasattr(self, '_download_btn'):
            self._download_btn.destroy()

        self._progress_frame.pack(fill="x", pady=(10, 0))
        self._progress_label.pack()
        self._progress_bar.pack()
        self._progress_bar.start()

        self._page_vars["model_downloading"].set(True)

        def download_thread():
            try:
                from faster_whisper import WhisperModel

                self._root.after(0, lambda: self._progress_label.config(text=f"正在下载 {model_size} 模型..."))

                # This will download the model
                model = WhisperModel(model_size, device="cpu", compute_type="int8")

                # Success
                self._root.after(0, self._download_complete)

            except Exception as e:
                self._root.after(0, lambda: self._download_failed(str(e)))

        threading.Thread(target=download_thread, daemon=True).start()

    def _download_complete(self) -> None:
        """Handle model download completion."""
        self._progress_bar.stop()
        self._progress_frame.pack_forget()

        self._page_vars["model_downloading"].set(False)
        self._page_vars["model_downloaded"].set(True)

        tk.Label(
            self._download_frame,
            text="✓ 模型下载完成",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#4CAF50",
        ).pack()

    def _download_failed(self, error: str) -> None:
        """Handle model download failure."""
        self._progress_bar.stop()

        self._page_vars["model_downloading"].set(False)

        for widget in self._progress_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self._progress_frame,
            text=f"下载失败: {error}",
            font=("Microsoft YaHei UI", 9),
            bg="white",
            fg="#f44336",
        ).pack()

        tk.Button(
            self._progress_frame,
            text="重试",
            font=("Microsoft YaHei UI", 9),
            bg="#2196F3",
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self._start_model_download,
        ).pack(pady=5)

    def _verify_llm_config(self) -> None:
        """Verify LLM configuration."""
        # Clear frame and show loading
        for widget in self._llm_verify_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self._llm_verify_frame,
            text="正在验证配置...",
            font=("Microsoft YaHei UI", 11),
            bg="white",
            fg="#2196F3",
        ).pack(pady=30)

        self._root.update()

        base_url = self._page_vars["llm_base_url"].get().strip()
        api_key = self._page_vars["llm_api_key"].get().strip()
        model = self._page_vars["llm_model"].get().strip()

        verified = False
        message = ""
        can_continue = False  # Whether user can proceed despite verification failure

        try:
            if not base_url or not api_key or not model:
                message = "✗ 配置不完整\n\n请填写 API 地址、密钥和模型名称"
                can_continue = False
            else:
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(
                        f"{base_url.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "Hi"}],
                            "max_tokens": 10,
                        },
                    )

                if response.status_code == 200:
                    verified = True
                    message = "✓ LLM 配置正常\n✓ API 连接验证通过"
                    can_continue = True
                elif response.status_code == 401:
                    message = "✗ API 密钥错误\n\n请检查密钥是否正确"
                    can_continue = False
                elif response.status_code == 404:
                    message = "✗ 模型不存在\n\n请检查模型名称是否正确"
                    can_continue = False
                else:
                    try:
                        error_detail = response.json().get("error", {}).get("message", str(response.status_code))
                    except Exception:
                        error_detail = str(response.status_code)
                    message = f"✗ API 返回错误: {error_detail}\n\n可能是网络问题，您可以稍后重试，或点击「继续」稍后再验证。"
                    can_continue = True

        except httpx.TimeoutException:
            message = "✗ 连接超时\n\n可能是网络问题，您可以稍后重试，或点击「继续」稍后再验证。"
            can_continue = True
        except httpx.ConnectError:
            message = "✗ 无法连接到服务器\n\n请检查 API 地址是否正确，或点击「继续」稍后再验证。"
            can_continue = True
        except Exception as e:
            message = f"✗ 验证失败：{e}\n\n可能是网络问题，您可以稍后重试，或点击「继续」稍后再验证。"
            can_continue = True

        # Update UI
        for widget in self._llm_verify_frame.winfo_children():
            widget.destroy()

        if verified:
            tk.Label(
                self._llm_verify_frame,
                text=message,
                font=("Microsoft YaHei UI", 11),
                bg="white",
                fg="#4CAF50",
                justify="left",
            ).pack(pady=20)

            tk.Label(
                self._llm_verify_frame,
                text="配置完成！可以继续下一步了。",
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg="#666666",
            ).pack()

            self._page_vars["llm_verified"] = True
        else:
            tk.Label(
                self._llm_verify_frame,
                text=message,
                font=("Microsoft YaHei UI", 11),
                bg="white",
                fg="#f44336" if not can_continue else "#FF9800",
                justify="left",
            ).pack(pady=20)

            btn_frame = tk.Frame(self._llm_verify_frame, bg="white")
            btn_frame.pack(pady=10)

            tk.Button(
                btn_frame,
                text="重试",
                font=("Microsoft YaHei UI", 10),
                bg="#2196F3",
                fg="white",
                relief="flat",
                cursor="hand2",
                command=self._verify_llm_config,
            ).pack(side="left", padx=5)

            if can_continue:
                tk.Button(
                    btn_frame,
                    text="继续",
                    font=("Microsoft YaHei UI", 10),
                    bg="#4CAF50",
                    fg="white",
                    relief="flat",
                    cursor="hand2",
                    command=lambda: self._page_vars.update({"llm_verified": True}) or self._on_next(),
                ).pack(side="left", padx=5)

            self._page_vars["llm_verified"] = False

    def _page_persona_selection(self, parent: tk.Frame) -> None:
        """Show persona selection page (Page 5)."""
        frame = tk.Frame(parent, bg="white", padx=40, pady=30)
        frame.pack(fill="both", expand=True)

        tk.Label(
            frame,
            text="选择人格面具",
            font=("Microsoft YaHei UI", 14, "bold"),
            bg="white",
            fg="#333333",
        ).pack(pady=(0, 10))

        tk.Label(
            frame,
            text="选择您想要激活的人格面具（录音时可见）",
            font=("Microsoft YaHei UI", 10),
            bg="white",
            fg="#666666",
        ).pack(pady=(0, 20))

        # Load available personas
        from untype.config import load_personas, get_personas_dir
        import json

        personas_dir = get_personas_dir()
        personas_list = []

        if personas_dir.is_dir():
            for path in sorted(personas_dir.glob("*.json")):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and all(k in data for k in ("id", "name", "icon")):
                        # Default to active for old files without the field
                        personas_list.append({
                            "id": data["id"],
                            "name": data["name"],
                            "icon": data["icon"],
                            "active": data.get("active", True),
                        })
                except Exception:
                    continue

        # Store persona checkbox variables
        if "persona_checkboxes" not in self._page_vars:
            self._page_vars["persona_checkboxes"] = {}
            self._page_vars["persona_ids"] = []

        # Create scrollable frame for personas
        container = tk.Frame(frame, bg="white")
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        scrollable_frame = tk.Frame(canvas, bg="white")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add persona checkboxes
        if personas_list:
            for i, persona in enumerate(personas_list):
                row = tk.Frame(scrollable_frame, bg="white", padx=10, pady=5)
                row.pack(fill="x")

                checkbox_var = self._create_boolean_var(value=persona["active"])
                self._page_vars["persona_checkboxes"][persona["id"]] = checkbox_var
                self._page_vars["persona_ids"].append(persona["id"])

                chk = tk.Checkbutton(
                    row,
                    variable=checkbox_var,
                    bg="white",
                    activebackground="white",
                    font=("Microsoft YaHei UI", 10),
                )
                chk.pack(side="left", padx=(0, 10))

                tk.Label(
                    row,
                    text=f"{persona['icon']} {persona['name']}",
                    font=("Microsoft YaHei UI", 10),
                    bg="white",
                    fg="#333333",
                ).pack(side="left")

                # Mouse wheel binding
                canvas.bind_all("<MouseWheel>", lambda e, c=canvas: c.yview_scroll(int(-1*(e.delta/120)), "units"))
        else:
            tk.Label(
                scrollable_frame,
                text="暂无可用的人格面具",
                font=("Microsoft YaHei UI", 10),
                bg="white",
                fg="#888888",
            ).pack(pady=20)

        # Help text at bottom
        help_frame = tk.Frame(frame, bg="#f0f8ff", padx=15, pady=12)
        help_frame.pack(fill="x", pady=(15, 0))

        tk.Label(
            help_frame,
            text="💡 提示",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg="#f0f8ff",
            fg="#333333",
        ).pack(anchor="w")

        tk.Label(
            help_frame,
            text="• 只有激活的人格面具才会在录音时显示\n• 之后可以在设置中添加自定义人格面具",
            font=("Microsoft YaHei UI", 9),
            bg="#f0f8ff",
            fg="#666666",
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

    # ------------------------------------------------------------------
    # Page registration
    # ------------------------------------------------------------------

    def _init_pages(self) -> None:
        """Initialize all pages."""
        self._pages = [
            self._page_welcome,
            self._page_stt_selection,
            self._get_page_2(),  # Dynamic based on selection
            self._page_stt_verify,
            self._page_llm_config,
            self._page_llm_verify,
            self._page_persona_selection,
            self._page_quick_start,
        ]

    def _show_page(self, page_num: int) -> None:
        """Override to handle dynamic page 2."""
        self._current_page = page_num

        # Rebuild pages list when entering page 2 to refresh it based on selection
        if page_num == 2:
            self._pages[2] = self._get_page_2()

        # Clear content frame
        for widget in self._content_frame.winfo_children():
            widget.destroy()

        # Show current page
        self._pages[page_num](self._content_frame)

        # Update navigation buttons
        self._back_btn.config(state="normal" if page_num > 0 else "disabled")

        is_last = page_num == len(self._pages) - 1
        self._next_btn.config(text="完成" if is_last else "下一步 >")


def run_setup_wizard(config: AppConfig, on_complete: Callable[[AppConfig], None]) -> None:
    """Run the setup wizard."""
    import tkinter as tk
    import gc
    import time

    # Store the wizard's root for cleanup
    wizard_root = None

    class CleanupWizard(SetupWizard):
        def run(self):
            nonlocal wizard_root
            self._create_ui()
            self._init_pages()
            self._show_page(0)
            wizard_root = self._root
            self._root.mainloop()

    wizard = CleanupWizard(config, on_complete)
    wizard.run()

    # IMPORTANT: Thorough cleanup after wizard closes
    # This is critical because overlay will create a new Tk() in a different thread

    # Step 1: Destroy the wizard's root if it still exists
    if wizard_root is not None:
        try:
            wizard_root.destroy()
        except Exception:
            pass
        wizard_root = None

    # Step 2: Clear all tkinter internal references
    try:
        # Clear default root
        if hasattr(tk, '_default_root'):
            tk._default_root = None
        # Clear misc root
        if hasattr(tk, '_misc') and hasattr(tk._misc, '_root'):
            tk._misc._root = None
        # Reset support flag
        if hasattr(tk, '_support_default_root'):
            tk._support_default_root = True
    except Exception:
        pass

    # Step 3: Force garbage collection multiple times
    # This is important because Tcl objects have circular references
    for _ in range(3):
        gc.collect()
        time.sleep(0.05)  # Small pause between collections

    # Final delay before main app starts
    time.sleep(0.2)
