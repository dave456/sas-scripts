#
# Star Reducer using Bill Blanshan's pixelmath formulas
#
# SPDX-License-Identifier: GPL-3.0
# Author: Dave Lindner (c) 2026 lindner234 <AT> gmail
#

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
SCRIPT_NAME     = "Star Reducer"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""   # optional

SOFT_ITER_TYPE = 0
MODERATE_ITER_TYPE = 1
STRONG_ITER_TYPE = 2

class StarReducerWindow(QDialog):
    def __init__(self, ctx, parent):
        """ Constructor for our UI class """
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Bill Blanshan's Star Reducer")
        self.setFixedWidth(550)
        self.starless_file_path = ""
        self.CreateWidgets()

    def CreateWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a box and frame for reduction options
        options_box = QGroupBox(" Reduction Options ")    
        options_frame = QVBoxLayout()
        options_box.setLayout(options_frame)
        options_box.setContentsMargins(8, 23, 8, 13)

        xfer_row = QHBoxLayout()
        self.xfer_btn = QRadioButton("Star reduction using midtones transfer function")
        self.xfer_btn.setChecked(True)
        self.xfer_btn.toggled.connect(self.OnToggled)
        xfer_row.addWidget(self.xfer_btn)
        options_box.layout().addLayout(xfer_row)
        
        halo_row = QHBoxLayout()
        self.halo_btn = QRadioButton("Star reduction using midtones transfer function (preserve halos)")
        self.halo_btn.toggled.connect(self.OnToggled)
        halo_row.addWidget(self.halo_btn)
        options_box.layout().addLayout(halo_row)

        star_row = QHBoxLayout()
        self.star_btn = QRadioButton("Iterative star reduction")
        self.star_btn.toggled.connect(self.OnToggled)
        star_row.addWidget(self.star_btn)
        options_box.layout().addLayout(star_row)
        
        layout.addWidget(options_box)
        layout.addSpacing(10)

        # Create box and frame for MTF slider
        params_box = QGroupBox(" MTF Adjustment ")
        params_frame = QVBoxLayout()
        params_box.setLayout(params_frame)
        params_box.setContentsMargins(8, 23, 8, 13)

        strength_row = QHBoxLayout()
        reduce_label = QLabel("Increase")
        reduce_label.setFixedWidth(50)
        strength_row.addWidget(reduce_label)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(80)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.setTickInterval(50)
        strength_row.addWidget(self.strength_slider, 1)
        increase_label = QLabel("Reduce")
        increase_label.setFixedWidth(50)
        strength_row.addWidget(increase_label)
        params_box.layout().addLayout(strength_row)

        layout.addWidget(params_box)
        layout.addSpacing(10)

        # Create box and frame for iterator star reduction parameters
        iter_params_box = QGroupBox(" Iterative Reduction Parameters ")
        iter_params_frame = QVBoxLayout()
        iter_params_box.setLayout(iter_params_frame)
        iter_params_box.setContentsMargins(8, 23, 8, 13)

        iter_params_row = QHBoxLayout()
        iter_params_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        iter_params_row.setSpacing(10)

        iter_type_label = QLabel("Strength:")
        iter_type_label.setFixedWidth(47)
        iter_params_row.addWidget(iter_type_label)
        self.iter_type = QComboBox()
        self.iter_type.addItem("Soft")
        self.iter_type.addItem("Moderate")
        self.iter_type.addItem("Strong")
        self.iter_type.setCurrentIndex(1)
        self.iter_type.setFixedWidth(100)
        self.iter_type.setEnabled(False)
        iter_params_row.addWidget(self.iter_type)

        iter_label = QLabel("Iterations:")
        iter_label.setFixedWidth(55)
        iter_params_row.addWidget(iter_label)
        self.iter_cnt = QSpinBox()
        self.iter_cnt.setFixedWidth(60)
        self.iter_cnt.setMinimum(1)
        self.iter_cnt.setMaximum(3)
        self.iter_cnt.setValue(1)
        self.iter_cnt.setEnabled(False)
        iter_params_row.addWidget(self.iter_cnt, 1)
        iter_params_row.addSpacing(20)

        iter_params_box.layout().addLayout(iter_params_row)
        layout.addWidget(iter_params_box)
        layout.addSpacing(10)

        starless_path_box = QGroupBox(" Starless Image Path ")
        starless_path_frame = QVBoxLayout()
        starless_path_box.setLayout(starless_path_frame)
        starless_path_box.setContentsMargins(10, 25, 10, 15)

        starless_row = QHBoxLayout()
        self.starless_path_edit = QLineEdit()
        starless_row.addWidget(self.starless_path_edit, 1)
        starless_btn = QPushButton("Select")
        starless_btn.clicked.connect(self.OnSelectStarless)
        starless_row.addWidget(starless_btn)
        starless_path_box.layout().addLayout(starless_row)

        starless_info_row = QHBoxLayout()
        self.starless_info = QLabel("")
        starless_info_row.addWidget(self.starless_info)
        self.starless_info.setText("  No starless image selected.")
        starless_path_box.layout().addLayout(starless_info_row)

        layout.addWidget(starless_path_box)
        layout.addSpacing(10)

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
    
    def OnToggled(self):
        """Gray out options that are not relevant to the selected reduction method"""
        if self.xfer_btn.isChecked():
            self.strength_slider.setEnabled(True)
            self.iter_cnt.setEnabled(False)
            self.iter_type.setEnabled(False)
        elif self.star_btn.isChecked():
            self.strength_slider.setEnabled(False)
            self.iter_cnt.setEnabled(True)
            self.iter_type.setEnabled(True)
        elif self.halo_btn.isChecked():
            self.strength_slider.setEnabled(True)
            self.iter_cnt.setEnabled(False)
            self.iter_type.setEnabled(False)

    def OnSelectStarless(self):
        """Open file dialog to select starless image"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select starless image", "", "FITS files (*.fits);;All files (*.*)")
        if file_path:
            self.starless_path_edit.setText(os.path.basename(file_path))
            self.starless_file_path = file_path
            self.starless_info.setText("  Starless image selected.")

    def OnApply(self):
        """Apply button callback"""
        if not self.starless_file_path:
            QMessageBox.warning(self, "No starless image", "Please select a starless image before applying star reduction.")
            return
        self.ReduceStars()

    def ReduceStars(self):
        try:
            img = self.ctx.get_image()

            # SAS gives HWC layout; FITS/numpy math expects CHW — only transpose if needed
            needs_transpose = img.ndim == 3 and img.shape[2] in (1, 3)
            is_color = img.ndim == 3 and (img.shape[2] == 3 or img.shape[0] == 3)

            if needs_transpose and is_color:
                img = np.transpose(img, (2, 0, 1))

            with fits.open(self.starless_file_path) as hdul:
                starless = hdul[0].data
                if starless.dtype != np.float32:
                    starless = np.array(starless, dtype=np.float32)

            # we can't allow 0 or 1 for reduction value, so we clamp to 0.01 and 0.99
            rv = self.strength_slider.value() / 100.0
            if rv == 1.0:
                rv = 0.99
            if rv == 0.0:
                rv = 0.01
    
            # MTF reduction
            if self.xfer_btn.isChecked():
                # ~((~mtf(rv, img) / ~mtf(rv, starless)) * ~starless)
                with np.errstate(divide='ignore', invalid='ignore'):
                    new_img = inv((inv(mtf(rv, img)) / inv(mtf(rv, starless))) * inv(starless))

            # MTF reduction with halo preservation
            # I don't think this works particularly well \o/ but its here for completeness
            elif self.halo_btn.isChecked():
                # (~(~img_data / ~starless) - ~(~mtf(rv, img_data) / ~mtf(rv, starless))) * ~starless
                h1 = (inv(inv(img) / inv(starless)) - inv(inv(mtf(rv, img)) / inv(mtf(rv, starless)))) * inv(starless)

                # ~(~img_data / ~starless) - ~(~mtf(rv, img_data) / ~mtf(rv, starless))
                h2 = inv(inv(img) / inv(starless)) - inv(inv(mtf(rv, img) / inv(mtf(rv, starless))))
                new_img = img * inv((h1 + h2) / 2)

            elif self.star_btn.isChecked():
                with np.errstate(divide='ignore', invalid='ignore'):
                    # img * ~(~(max(0, min(1, starless / img))) * ~img)
                    s1 = img * inv(inv(np.maximum(0, np.minimum(1, starless / img))) * inv(img))

                    # max(s1, (img * s1) + (s1 * ~s1))
                    s2 = np.maximum(s1, (img * s1) + (s1 * inv(s1)))

                    if self.iter_cnt.value() >= 2:
                        # s1 * ~(~(max(0, min(1, starless / s1))) * ~s1)
                        s3 = s1 * inv(inv(np.maximum(0, np.minimum(1, starless / s1))) * inv(s1))

                        # max(s3, (img * s3) + (s3 * ~s3))
                        s4 = np.maximum(s3, (img * s3) + (s3 * inv(s3)))

                    if self.iter_cnt.value() == 3:
                        # s3 * ~(~(max(0, min(1, starless / s3))) * ~s3)
                        s5 = s3 * inv(inv(np.maximum(0, np.minimum(1, starless / s3))) * inv(s3))

                        # max(s5, (img * s5) + (s5 * ~s5))
                        s6 = np.maximum(s5, (img * s5) + (s5 * inv(s5)))

                    if self.iter_type.currentIndex() == STRONG_ITER_TYPE:
                        match self.iter_cnt.value():
                            case 1:
                                new_img = s1
                            case 2:
                                new_img = s3
                            case 3:
                                new_img = s5
                    if self.iter_type.currentIndex() == MODERATE_ITER_TYPE:
                        match self.iter_cnt.value():
                            case 1:
                                new_img = s2
                            case 2:
                                new_img = s4
                            case 3:
                                new_img = s6

                    # mean(img - (img - iif(I==1, s2, iif(I==2, s4, s6))), img * ~(img - iif(I==1, s2, iif(I==2, s4, s6))))
                    if self.iter_type.currentIndex() == SOFT_ITER_TYPE:
                        match self.iter_cnt.value():
                            case 1:
                                new_img = ((img - (img - s2)) + (img * inv(img - s2))) / 2
                            case 2:
                                new_img = ((img - (img - s4)) + (img * inv(img - s4))) / 2
                            case 3:
                                new_img = ((img - (img - s6)) + (img * inv(img - s6))) / 2

            if needs_transpose and is_color:
                new_img = np.transpose(new_img, (1, 2, 0))
            self.ctx.set_image(new_img, step_name="Star reduction")
            self.ctx.log("Star reduction applied.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during star reduction:\n{str(e)}")
            self.ctx.log(f"Error during star reduction: {str(e)}")

    def OnHelp(self):
        QMessageBox.information(self, "Help", "This tool applies Bill Blanshan's star reduction methods to an image.\n\n"
            "1. Generate and/or select a starless image using the tool of your choice.\n"
            "2. Choose the reduction method (MTF or iterative).\n"
            "3. Adjust parameters as needed.\n"
            "4. Click Apply to perform the star reduction on the current image.\n\n"
            "Note: The image should already be stretched before applying star reduction.")

def inv(img):
    """
    Pixelmath invert function (~)

    :param img: numpy image to invert
    """
    return 1.0 - img

def mtf(m, img, clipResult = False):
    """
    Pixelmath Midtones transfer function (mtf)
    
    :param m: midtones value
    :param img: numpy image to perform midtones transfer on
    :param clipResult: optionally clip the result, default is false
    """
    if m == 0.5:
        return img
    
    clipped = np.clip(img, 0, 1)
    res = ((m - 1) * clipped) / (((2 * m - 1) * clipped) - m)

    if clipResult:
        return np.clip(res, 0, 1)
    else:
        return res

def run(ctx):
    """
    SASpro entry point.
    """
    w = StarReducerWindow(ctx, parent=ctx.app)
    w.setModal(False)
    w.setWindowModality(Qt.WindowModality.NonModal)
    w.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_star_reducer_window", w)
    except Exception:
        pass

    return w
