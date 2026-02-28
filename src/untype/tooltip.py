"""Simple tooltip implementation for tkinter widgets."""

import tkinter as tk


class Tooltip:
    """Attach a tooltip to a widget.

    Example:
        from untype.tooltip import Tooltip

        button = tk.Button(root, text="Click me")
        Tooltip(button, "This is a tooltip")
    """

    def __init__(self, widget: tk.Widget, text: str) -> None:
        """Initialize the tooltip.

        Args:
            widget: The widget to attach the tooltip to.
            text: The tooltip text to display.
        """
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None

        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        """Show the tooltip window."""
        if self.tip_window or not self.widget.winfo_containing(
            self.widget.winfo_pointerx(), self.widget.winfo_pointery()
        ):
            return

        x = self.widget.winfo_pointerx() + 25
        y = self.widget.winfo_pointery() + 20

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            bg="#ffffe0",
            fg="#000000",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 8),
            padx=4,
            pady=2,
        )
        label.pack()

        # Auto-hide after 5 seconds
        tw.after(5000, self.hide_tip)

    def hide_tip(self, event: tk.Event | None = None) -> None:  # type: ignore[type-arg]
        """Hide the tooltip window."""
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None
