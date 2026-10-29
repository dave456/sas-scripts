#
# Narrow band mixer for dual narrow band images (Ha + OIII)
#
# SPDX-License-Identifier: GPL-3.0
# Author: Dave Lindner (c) 2026 lindner234 <AT> gmail
#

import os
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QSlider, QComboBox
)
from PyQt6.QtCore import Qt
from astropy.io import fits

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Narrow Band Mixer"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""   # optional

OUTPUT_DOC_NAME = "NB-Blended"


class NbMixerWindow(QDialog):
    def __init__(self, ctx, parent):
        """ Constructor for our UI class """
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Dual Narrow Band Mixer")
        self.setFixedWidth(600)

        # initialize some member variables
        self.combos = {}   # component key -> source combo box
        # percentage of each component in each output channel (may exceed 100)
        self.mix = {
            "red": {"ha": 100, "oiii": 0},
            "green": {"ha": 50, "oiii": 50},
            "blue": {"ha": 0, "oiii": 100},
        }
        
        self.create_widgets()


    def create_widgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # ui constants
        LABEL_WIDTH = 20
        VALUE_LABEL_WIDTH = 70
        MAX_PERCENT = 200

        # create a box for our columns
        comps_box = QGroupBox(" Components ")
        comps_box.setLayout(QVBoxLayout())
    
        # add HA and OIII rows, each with a source combo (open views or a file)
        for comp, comp_name in (("ha", "Ha"), ("oiii", "OIII")):
            row = QHBoxLayout()
            label = QLabel(f"{comp_name}:")
            label.setFixedWidth(LABEL_WIDTH)
            row.addWidget(label)
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.combos[comp] = combo
            row.addWidget(combo, 1)
            file_btn = QPushButton("File...")
            file_btn.clicked.connect(lambda _checked=False, k=comp: self.on_select_file(k))
            row.addWidget(file_btn)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            comps_box.layout().addLayout(row)

        refresh_btn = QPushButton("Refresh views")
        refresh_btn.clicked.connect(self.refresh_views)
        comps_box.layout().addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.refresh_views()

        layout.addWidget(comps_box)
        layout.addSpacing(10)

        # Add Blender component
        blender_box = QGroupBox(" Mixer ")
        blender_box.setLayout(QVBoxLayout())

        # one Ha slider and one OIII slider per output channel
        for channel in ("red", "green", "blue"):
            box = QGroupBox(f"{channel.capitalize()}  ")
            box.setLayout(QVBoxLayout())
            box.setFlat(True)
            blender_box.layout().addWidget(box)

            for comp, comp_name in (("ha", "Ha"), ("oiii", "OIII")):
                row = QHBoxLayout()
                label = QLabel(f"{comp_name}: {self.mix[channel][comp]}%")
                label.setFixedWidth(VALUE_LABEL_WIDTH)
                row.addWidget(label)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, MAX_PERCENT)
                slider.setValue(self.mix[channel][comp])
                slider.valueChanged.connect(
                    lambda value, c=channel, k=comp, n=comp_name, lbl=label:
                        self.on_slider_changed(c, k, n, lbl, value))
                row.addWidget(slider)
                box.layout().addLayout(row)

        layout.addWidget(blender_box)
        layout.addSpacing(10)

        # Blend button
        blend_btn = QPushButton("Blend")
        blend_btn.clicked.connect(self.on_blend)
        blend_btn.setFixedWidth(80)
        layout.addWidget(blend_btn, alignment=Qt.AlignmentFlag.AlignCenter)
    
    
    def refresh_views(self):
        """ Repopulate the source combos with open SASpro views, keeping any chosen file """
        views = [(v.get("name") or v.get("title") or str(v.get("uid")), v.get("uid"))
                 for v in self.ctx.list_views()]
        for combo in self.combos.values():
            current = combo.currentData()
            combo.clear()
            combo.addItem("-- select view or file --", None)
            for name, uid in views:
                combo.addItem(name, ("view", uid))
            if current and current[0] == "file":
                combo.addItem(os.path.basename(current[1]), current)
            index = combo.findData(current) if current else 0
            combo.setCurrentIndex(max(index, 0))

    def on_select_file(self, comp):
        """ File selection button callback """
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "FITS files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz *.fz2);;All files (*)")
        if path:
            combo = self.combos[comp]
            data = ("file", path)
            index = combo.findData(data)
            if index < 0:
                combo.addItem(os.path.basename(path), data)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

    def load_source(self, comp):
        """ Returns (pixel data, FITS header or None) from the selected view or file """
        kind, value = self.combos[comp].currentData()
        if kind == "file":
            with fits.open(value) as hdul:
                return hdul[0].data, hdul[0].header
        img = np.asarray(self.ctx.get_image_for(value))
        return (img.mean(axis=2) if img.ndim == 3 else img), None

    def on_slider_changed(self, channel, comp, comp_name, label, value):
        """ Any mixer slider changed """
        self.mix[channel][comp] = value
        label.setText(f"{comp_name}: {value}%")

    def on_blend(self):
        """ Blend button callback """
        if not all(c.currentData() for c in self.combos.values()):
            QMessageBox.warning(self, "Warning", "Please select both Ha and OIII views or files")
            return

        try:
            # Load images
            ha_data, ha_header = self.load_source("ha")
            oiii_data, _ = self.load_source("oiii")

            # Create RGB channels
            channels = [(self.mix[c]["ha"] / 100) * ha_data + (self.mix[c]["oiii"] / 100) * oiii_data
                        for c in ("red", "green", "blue")]

            # Stack to (3, height, width), then move to (height, width, 3) as SASpro expects
            combined_data = np.array(channels, dtype=np.float32)
            combined_data = np.transpose(combined_data, (1, 2, 0))

            # Clip values to the range [0, 1] to avoid pixels greater than 1.0, this will
            # likely make the image uglier at these points, but should be obvious and force
            # the user to adjust the mixer accordingly.
            np.minimum(combined_data, 1.0, out=combined_data)

            # use the Ha file's fits header; views have none
            header = ha_header if ha_header is not None else fits.Header()
            r, g, b = self.mix["red"], self.mix["green"], self.mix["blue"]
            header.add_history(f"NarrowBandMixer (Ha/OII) mix: R({r['ha']}%/{r['oiii']}%), "
                               f"G({g['ha']}%/{g['oiii']}%), B({b['ha']}%/{b['oiii']}%)")

            # update or open our blended view in SASpro
            if self.ctx.get_document(OUTPUT_DOC_NAME):
                self.ctx.set_image_for(OUTPUT_DOC_NAME, combined_data, step_name='Update NB Blend')
            else:
                self.ctx.open_new_document(combined_data, metadata=header, name=OUTPUT_DOC_NAME)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to blend images: {str(e)}")


def run(ctx):
    """
    SASpro entry point.
    """
    win = NbMixerWindow(ctx, parent=ctx.app)
    win.setModal(False)
    win.setWindowModality(Qt.WindowModality.NonModal)
    win.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_nb_mixer_window", win)
    except Exception:
        pass

    return win
