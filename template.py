import os
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QComboBox, QMessageBox, QGroupBox, QSlider, QRadioButton,
    QSpinBox, QFileDialog
)
from PyQt6.QtCore import Qt
from astropy.io import fits

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "My Dialog"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""   # optional

class TemplateWindow(QDialog):
    def __init__(self, ctx, parent):
        """ Constructor for our UI class """
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Template Dialog")
        self.setFixedWidth(550)
        self.CreateWidgets()

    def CreateWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Buttons
        button_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.OnApply)
        apply_btn.setFixedWidth(80)
        button_row.addWidget(apply_btn)

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.OnHelp)
        help_btn.setFixedWidth(80)
        button_row.addWidget(help_btn)

        layout.addLayout(button_row)

    def OnApply(self):
        return
    
    def OnHelp(self):
        return

def run(ctx):
    """
    SASpro entry point.
    """
    win = TemplateWindow(ctx, parent=ctx.app)
    win.setModal(False)
    win.setWindowModality(Qt.WindowModality.NonModal)
    win.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_template_window", win)
    except Exception:
        pass

    return win
