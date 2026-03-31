from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RecentResultEntry:
    """A lightweight session-scoped record of a produced result."""

    id: str
    created_at: datetime
    raw_text: str
    result_text: str
    mode: str
    status: str
    window_title: str = ""
    persona_id: str | None = None
    persona_name: str | None = None
    persona_icon: str | None = None

    @property
    def timestamp_label(self) -> str:
        return self.created_at.strftime("%H:%M:%S")

    @property
    def preview_text(self) -> str:
        text = " ".join(self.result_text.split())
        if len(text) <= 72:
            return text
        return text[:69] + "..."

    @property
    def persona_label(self) -> str:
        if self.persona_name and self.persona_icon:
            return f"{self.persona_icon} {self.persona_name}"
        if self.persona_name:
            return self.persona_name
        return "Default"
