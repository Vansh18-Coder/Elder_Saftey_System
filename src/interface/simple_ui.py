from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SimpleSafetyUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Elder Safety System")
        self.root.geometry("420x260")
        self.root.minsize(420, 260)

        self.status_var = tk.StringVar(value="NORMAL")
        self.confidence_var = tk.StringVar(value="Confidence: 0.00")

        ttk.Label(self.root, text="System Status", font=("Segoe UI", 12, "bold")).pack(pady=(12, 4))
        ttk.Label(self.root, textvariable=self.status_var, foreground="#1d4ed8", font=("Segoe UI", 18, "bold")).pack()
        ttk.Label(self.root, textvariable=self.confidence_var, font=("Segoe UI", 10)).pack(pady=(4, 12))

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=8)

        self.help_button = ttk.Button(button_frame, text="HELP", command=self._help_action)
        self.help_button.grid(row=0, column=0, padx=8)

        self.cancel_button = ttk.Button(button_frame, text="CANCEL", command=self._cancel_action)
        self.cancel_button.grid(row=0, column=1, padx=8)

        self.exit_button = ttk.Button(button_frame, text="EXIT", command=self.root.destroy)
        self.exit_button.grid(row=0, column=2, padx=8)

    def _help_action(self):
        self.status_var.set("HELP REQUESTED")
        self.confidence_var.set("Confidence: N/A")

    def _cancel_action(self):
        self.status_var.set("NORMAL")
        self.confidence_var.set("Confidence: 0.00")

    def set_status(self, status, confidence=None):
        self.status_var.set(status)
        if confidence is not None:
            self.confidence_var.set(f"Confidence: {confidence:.2f}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SimpleSafetyUI()
    app.run()
