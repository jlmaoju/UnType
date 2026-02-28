"""Persona Manager dialog — master-detail tkinter UI for managing personas."""

from __future__ import annotations

import json
import logging
import os
import re
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from untype.config import (
    Persona,
    delete_persona,
    get_personas_dir,
    load_personas,
    save_persona,
)
from untype.i18n import t

logger = logging.getLogger(__name__)

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class PersonaManagerDialog:
    """Master-detail dialog for creating, editing, deleting, importing and
    exporting persona JSON files.

    Follows the same pattern as :class:`~untype.tray.SettingsDialog`: creates
    its own :class:`tk.Tk` root and runs ``mainloop()`` on the calling thread.
    Call :meth:`show` from a dedicated daemon thread.
    """

    def __init__(self, on_changed: Callable[[], None] | None = None) -> None:
        self._on_changed = on_changed
        self._personas: list[Persona] = []
        self._selected_index: int = -1
        # The id of the persona currently loaded in the editor (before edits).
        self._editing_original_id: str | None = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        """Show the dialog (blocks until closed)."""
        root = tk.Tk()
        root.title(t("persona.title"))
        root.resizable(True, True)
        root.minsize(600, 420)
        root.attributes("-topmost", True)
        root.after(100, lambda: root.attributes("-topmost", False))
        root.protocol("WM_DELETE_WINDOW", lambda: self._on_close(root))

        self._root = root

        # Root grid: let outer frame fill the window.
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # -- Master-detail layout ----------------------------------------
        outer = ttk.Frame(root, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")

        # outer: column 0 (list) fixed, column 1 (editor) stretches.
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Left: persona list
        list_frame = ttk.LabelFrame(outer, text=t("persona.list.title"), padding=6)
        list_frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        list_frame.rowconfigure(0, weight=1)

        self._listbox = tk.Listbox(
            list_frame,
            width=22,
            height=18,
            exportselection=False,
            font=("TkDefaultFont", 10),
        )
        self._listbox.grid(row=0, column=0, sticky="nsew")
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self._listbox.yview)
        list_scroll.grid(row=0, column=1, sticky="ns")
        self._listbox.configure(yscrollcommand=list_scroll.set)

        # Right: editor (scrollable)
        editor_frame = ttk.LabelFrame(outer, text=t("persona.editor.title"), padding=6)
        editor_frame.grid(row=0, column=1, sticky="nsew")
        editor_frame.columnconfigure(1, weight=1)

        row = 0
        self._id_var = tk.StringVar(master=root)
        row = self._text_row(root, editor_frame, t("persona.id"), self._id_var, row)

        self._name_var = tk.StringVar(master=root)
        row = self._text_row(root, editor_frame, t("persona.name"), self._name_var, row)

        self._icon_var = tk.StringVar(master=root)
        row = self._text_row(root, editor_frame, t("persona.icon"), self._icon_var, row)

        self._model_var = tk.StringVar(master=root)
        row = self._text_row(root, editor_frame, t("persona.model"), self._model_var, row)

        self._temp_var = tk.StringVar(master=root)
        row = self._text_row(root, editor_frame, t("persona.temperature"), self._temp_var, row)

        self._maxtok_var = tk.StringVar(master=root)
        row = self._text_row(root, editor_frame, t("persona.max_tokens"), self._maxtok_var, row)

        # Insert prompt (multi-line)
        ttk.Label(editor_frame, text=t("persona.prompt_insert")).grid(
            row=row,
            column=0,
            sticky="nw",
            padx=(0, 8),
            pady=(6, 0),
        )
        insert_frame = ttk.Frame(editor_frame)
        insert_frame.grid(row=row, column=1, sticky="nsew", pady=(6, 0))
        insert_frame.columnconfigure(0, weight=1)
        insert_frame.rowconfigure(0, weight=1)
        editor_frame.rowconfigure(row, weight=1)
        self._insert_text = tk.Text(
            insert_frame,
            width=42,
            height=5,
            wrap="word",
            font=("TkDefaultFont", 9),
        )
        self._insert_text.grid(row=0, column=0, sticky="nsew")
        insert_scroll = ttk.Scrollbar(
            insert_frame,
            orient="vertical",
            command=self._insert_text.yview,
        )
        insert_scroll.grid(row=0, column=1, sticky="ns")
        self._insert_text.configure(yscrollcommand=insert_scroll.set)
        row += 1

        # Polish prompt (multi-line)
        ttk.Label(editor_frame, text=t("persona.prompt_polish")).grid(
            row=row,
            column=0,
            sticky="nw",
            padx=(0, 8),
            pady=(6, 0),
        )
        polish_frame = ttk.Frame(editor_frame)
        polish_frame.grid(row=row, column=1, sticky="nsew", pady=(6, 0))
        polish_frame.columnconfigure(0, weight=1)
        polish_frame.rowconfigure(0, weight=1)
        editor_frame.rowconfigure(row, weight=1)
        self._polish_text = tk.Text(
            polish_frame,
            width=42,
            height=5,
            wrap="word",
            font=("TkDefaultFont", 9),
        )
        self._polish_text.grid(row=0, column=0, sticky="nsew")
        polish_scroll = ttk.Scrollbar(
            polish_frame,
            orient="vertical",
            command=self._polish_text.yview,
        )
        polish_scroll.grid(row=0, column=1, sticky="ns")
        self._polish_text.configure(yscrollcommand=polish_scroll.set)
        row += 1

        # Active checkbox
        self._active_var = tk.BooleanVar(master=root)
        active_check = ttk.Checkbutton(
            editor_frame,
            text=t("persona.active"),
            variable=self._active_var,
        )
        active_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        row += 1

        # Save button
        save_btn_frame = ttk.Frame(editor_frame)
        save_btn_frame.grid(row=row, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(save_btn_frame, text=t("persona.save"), command=self._on_save).pack(side="right")
        row += 1

        # Status / error label
        self._status_var = tk.StringVar(master=root)
        self._status_label = ttk.Label(
            editor_frame,
            textvariable=self._status_var,
            foreground="red",
            wraplength=320,
        )
        self._status_label.grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # -- Bottom toolbar -----------------------------------------------
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Button(toolbar, text=t("persona.new"), command=self._on_new).pack(side="left")
        ttk.Button(toolbar, text=t("persona.delete"), command=self._on_delete).pack(
            side="left",
            padx=(6, 0),
        )
        ttk.Button(toolbar, text=t("persona.open_folder"), command=self._on_open_folder).pack(
            side="left",
            padx=(6, 0),
        )
        ttk.Button(toolbar, text=t("persona.export"), command=self._on_export).pack(
            side="right",
        )
        ttk.Button(toolbar, text=t("persona.import"), command=self._on_import).pack(
            side="right",
            padx=(0, 6),
        )

        # -- Load data ----------------------------------------------------
        self._refresh_list()
        self._clear_editor()

        # Centre window
        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() - w) // 2
        y = (root.winfo_screenheight() - h) // 2
        root.geometry(f"+{x}+{y}")

        root.mainloop()

    # ------------------------------------------------------------------ #
    # Editor helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _text_row(
        master: tk.Tk,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        row: int,
    ) -> int:
        ttk.Label(parent, text=label).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        ttk.Entry(parent, textvariable=var, width=36).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=2,
        )
        return row + 1

    def _clear_editor(self) -> None:
        self._editing_original_id = None
        self._id_var.set("")
        self._name_var.set("")
        self._icon_var.set("")
        self._model_var.set("")
        self._temp_var.set("")
        self._maxtok_var.set("")
        self._active_var.set(True)  # Default to active
        self._insert_text.delete("1.0", "end")
        self._polish_text.delete("1.0", "end")
        self._status_var.set("")

    def _load_into_editor(self, persona: Persona) -> None:
        self._editing_original_id = persona.id
        self._id_var.set(persona.id)
        self._name_var.set(persona.name)
        self._icon_var.set(persona.icon)
        self._model_var.set(persona.model)
        self._temp_var.set(str(persona.temperature) if persona.temperature is not None else "")
        self._maxtok_var.set(str(persona.max_tokens) if persona.max_tokens is not None else "")
        self._active_var.set(persona.active)
        self._insert_text.delete("1.0", "end")
        self._insert_text.insert("1.0", persona.prompt_insert)
        self._polish_text.delete("1.0", "end")
        self._polish_text.insert("1.0", persona.prompt_polish)
        self._status_var.set("")

    def _editor_to_persona(self) -> Persona | None:
        """Read editor fields and validate.  Returns None on validation error."""
        pid = self._id_var.get().strip()
        name = self._name_var.get().strip()
        icon = self._icon_var.get().strip()

        if not pid:
            self._status_var.set(t("persona.error.id_required"))
            return None
        if not _ID_PATTERN.match(pid):
            self._status_var.set(t("persona.error.id_format"))
            return None
        # Check collision with other personas (not self).
        for p in self._personas:
            if p.id == pid and pid != self._editing_original_id:
                self._status_var.set(t("persona.error.id_exists", id=pid))
                return None
        if not name:
            self._status_var.set(t("persona.error.name_required"))
            return None
        if not icon:
            self._status_var.set(t("persona.error.icon_required"))
            return None

        # Optional: temperature
        temp_str = self._temp_var.get().strip()
        temperature: float | None = None
        if temp_str:
            try:
                temperature = float(temp_str)
                if not 0.0 <= temperature <= 2.0:
                    self._status_var.set(t("persona.error.temp_range"))
                    return None
            except ValueError:
                self._status_var.set(t("persona.error.temp_number"))
                return None

        # Optional: max_tokens
        maxtok_str = self._maxtok_var.get().strip()
        max_tokens: int | None = None
        if maxtok_str:
            try:
                max_tokens = int(maxtok_str)
                if max_tokens <= 0:
                    self._status_var.set(t("persona.error.maxtok_positive"))
                    return None
            except ValueError:
                self._status_var.set(t("persona.error.maxtok_positive"))
                return None

        return Persona(
            id=pid,
            name=name,
            icon=icon,
            active=self._active_var.get(),
            prompt_polish=self._polish_text.get("1.0", "end-1c"),
            prompt_insert=self._insert_text.get("1.0", "end-1c"),
            model=self._model_var.get().strip(),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # ------------------------------------------------------------------ #
    # List management
    # ------------------------------------------------------------------ #

    def _refresh_list(self) -> None:
        self._personas = load_personas()
        self._listbox.delete(0, "end")
        for p in self._personas:
            self._listbox.insert("end", f"{p.icon} {p.name}")
        self._selected_index = -1

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #

    def _on_list_select(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self._selected_index:
            return
        self._selected_index = idx
        self._load_into_editor(self._personas[idx])

    def _on_save(self) -> None:
        persona = self._editor_to_persona()
        if persona is None:
            return

        # If the ID changed from the original, delete the old file.
        if self._editing_original_id is not None and self._editing_original_id != persona.id:
            delete_persona(self._editing_original_id)

        save_persona(persona)
        self._status_var.set("")
        logger.info("Persona '%s' saved", persona.id)

        # Refresh and re-select.
        self._refresh_list()
        for i, p in enumerate(self._personas):
            if p.id == persona.id:
                self._listbox.selection_set(i)
                self._listbox.see(i)
                self._selected_index = i
                self._editing_original_id = persona.id
                break

    def _on_new(self) -> None:
        # Deselect any current selection.
        self._listbox.selection_clear(0, "end")
        self._selected_index = -1

        self._clear_editor()
        self._id_var.set("new")
        self._name_var.set("New Persona")
        self._icon_var.set("\U0001f195")  # 🆕

    def _on_delete(self) -> None:
        if self._selected_index < 0:
            return
        persona = self._personas[self._selected_index]
        if not messagebox.askyesno(
            t("persona.delete_title"),
            t("persona.delete_confirm", icon=persona.icon, name=persona.name),
            parent=self._root,
        ):
            return
        delete_persona(persona.id)
        logger.info("Persona '%s' deleted", persona.id)
        self._refresh_list()
        self._clear_editor()

    def _on_import(self) -> None:
        files = filedialog.askopenfilenames(
            title=t("persona.import_title"),
            filetypes=[("JSON files", "*.json")],
            parent=self._root,
        )
        if not files:
            return

        personas_dir = get_personas_dir()
        personas_dir.mkdir(parents=True, exist_ok=True)
        imported = 0
        existing_ids = {p.id for p in self._personas}

        for fpath in files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Import: failed to read %s: %s", fpath, exc)
                continue

            if not isinstance(data, dict):
                continue
            if not all(k in data for k in ("id", "name", "icon")):
                logger.warning("Import: %s missing required fields — skipping", fpath)
                continue

            pid = data["id"]
            if pid in existing_ids:
                if not messagebox.askyesno(
                    t("persona.import_overwrite_title"),
                    t("persona.import_overwrite", id=pid),
                    parent=self._root,
                ):
                    continue

            # Write as <id>.json
            dest = personas_dir / f"{pid}.json"
            try:
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                imported += 1
            except OSError as exc:
                logger.warning("Import: failed to write %s: %s", dest, exc)

        self._refresh_list()
        self._status_var.set(t("persona.import_success", count=imported))

    def _on_export(self) -> None:
        if self._selected_index < 0:
            return
        persona = self._personas[self._selected_index]
        path = filedialog.asksaveasfilename(
            title=t("persona.export_title"),
            defaultextension=".json",
            initialfile=f"{persona.id}.json",
            filetypes=[("JSON files", "*.json")],
            parent=self._root,
        )
        if not path:
            return
        data = asdict(persona)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self._status_var.set(t("persona.export_success", file=Path(path).name))
        except OSError as exc:
            self._status_var.set(f"Export failed: {exc}")

    def _on_open_folder(self) -> None:
        personas_dir = get_personas_dir()
        personas_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(personas_dir)

    def _on_close(self, root: tk.Tk) -> None:
        root.destroy()
        if self._on_changed is not None:
            try:
                self._on_changed()
            except Exception:
                logger.exception("Error in on_changed callback")
