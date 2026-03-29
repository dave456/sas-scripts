# Banding reduction script for SASpro
# SPDX-License-Identifier: GPL-3.0
# Author: Dave Lindner (c) 2025 lindner234 <AT> gmail
"""
This script provides banding reduction for astronomical images.
"""

import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QMessageBox, QGroupBox, QSlider, QRadioButton,
    QSpinBox
)
from PyQt6.QtCore import Qt
from astropy.io import fits

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Remove Bandling"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""

class BandingReductionWindow(QDialog):
    def __init__(self, ctx, parent):
        """ Constructor for our UI class """
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Banding Reduction")
        self.setFixedWidth(525)
        self.CreateWidgets()

    def CreateWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a box and frame for reduction options
        options_box = QGroupBox(" Algorithm ")    
        options_frame = QVBoxLayout()
        options_box.setLayout(options_frame)
        options_box.setContentsMargins(8, 23, 8, 13)

        siril_row = QHBoxLayout()
        self.median_btn = QRadioButton("Sigma-clipped row medians, single pass")
        self.median_btn.setChecked(True)
        self.median_btn.toggled.connect(self.OnToggled)
        siril_row.addWidget(self.median_btn)
        options_box.layout().addLayout(siril_row)
        
        pi_row = QHBoxLayout()
        self.means_btn = QRadioButton("Sigma-clipped row means, multi-pass convergence")
        self.means_btn.toggled.connect(self.OnToggled)
        pi_row.addWidget(self.means_btn)
        options_box.layout().addLayout(pi_row)

        linear_row = QHBoxLayout()
        self.linear_btn = QRadioButton("Linear pattern subtraction (à trous wavelet)")
        self.linear_btn.toggled.connect(self.OnToggled)
        linear_row.addWidget(self.linear_btn)
        options_box.layout().addLayout(linear_row)
        
        layout.addWidget(options_box)
        layout.addSpacing(10)

        params_box = QGroupBox(" Parameters ")
        params_frame = QVBoxLayout()
        params_box.setLayout(params_frame)
        params_box.setContentsMargins(8, 23, 8, 13)

        strength_row = QHBoxLayout()
        strength_label = QLabel("Strength:")
        strength_label.setFixedWidth(60)
        strength_row.addWidget(strength_label)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(1, 100)
        self.strength_slider.setValue(50)
        self.strength_value_label = QLabel(f"{self.strength_slider.value()}%")
        self.strength_slider.valueChanged.connect(lambda v: self.strength_value_label.setText(f"{v}%"))

        strength_row.addWidget(self.strength_slider)
        strength_row.addWidget(self.strength_value_label)
        params_box.layout().addLayout(strength_row)

        sigma_row = QHBoxLayout()
        sigma_label = QLabel("Sigma:")
        sigma_label.setFixedWidth(60)
        sigma_row.addWidget(sigma_label)
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(1, 60)
        self.sigma_slider.setValue(30)
        self.sigma_value_label = QLabel(f"{self.sigma_slider.value()/10}σ")
        self.sigma_slider.valueChanged.connect(lambda v: self.sigma_value_label.setText(f"{v/10}σ"))

        sigma_row.addWidget(self.sigma_slider)
        sigma_row.addWidget(self.sigma_value_label)
        params_box.layout().addLayout(sigma_row)

        num_passes_row = QHBoxLayout()
        num_passes_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        passes_label = QLabel("Passes:")
        passes_label.setFixedWidth(60)
        self.num_passes = QSpinBox()
        self.num_passes.setFixedWidth(60)
        self.num_passes.setMinimum(1)
        self.num_passes.setMaximum(3)
        self.num_passes.setValue(1)
        self.num_passes.setEnabled(False)

        num_passes_row.addWidget(passes_label)
        num_passes_row.addWidget(self.num_passes, 1)
        params_box.layout().addLayout(num_passes_row)

        num_wavelets_row = QHBoxLayout()
        num_wavelets_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        wavelets_label = QLabel("Wavelet:")
        wavelets_label.setFixedWidth(60)
        self.num_wavelets = QSpinBox()
        self.num_wavelets.setFixedWidth(60)
        self.num_wavelets.setMinimum(1)
        self.num_wavelets.setMaximum(5)
        self.num_wavelets.setValue(2)
        self.num_wavelets.setEnabled(False)
        self.wavelet_desc_label = QLabel(" (1 ≈ 2px, 2 ≈ 4px, 3 ≈ 8px, etc.)")
        self.wavelet_desc_label.setStyleSheet("color: gray;")

        num_wavelets_row.addWidget(wavelets_label)
        num_wavelets_row.addWidget(self.num_wavelets, 1)
        num_wavelets_row.addWidget(self.wavelet_desc_label)
        params_box.layout().addLayout(num_wavelets_row)

        banding_direction_row = QHBoxLayout()
        banding_direction_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        banding_label = QLabel("Banding:")
        banding_label.setFixedWidth(60)
        self.banding_direction = QComboBox()
        self.banding_direction.addItems(["Horizontal", "Vertical"])

        banding_direction_row.addWidget(banding_label)
        banding_direction_row.addWidget(self.banding_direction)
        params_box.layout().addLayout(banding_direction_row)

        layout.addWidget(params_box)
        layout.addSpacing(10)

        # Buttons
        button_row = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.OnApply)
        self.apply_btn.setFixedWidth(80)
        button_row.addWidget(self.apply_btn)

        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.OnHelp)
        help_btn.setFixedWidth(80)
        button_row.addWidget(help_btn)

        layout.addLayout(button_row)

    def OnToggled(self):
        """Gray out options that are not relevant to the selected banding reduction method"""
        if self.median_btn.isChecked():
            self.num_passes.setEnabled(False)
            self.num_wavelets.setEnabled(False)
        elif self.means_btn.isChecked():
            self.num_passes.setEnabled(True)
            self.num_wavelets.setEnabled(False)
        elif self.linear_btn.isChecked():
            self.num_passes.setEnabled(False)
            self.num_wavelets.setEnabled(True)
        return

    def OnApply(self):
        self.apply_btn.setEnabled(False)
        img = self.ctx.get_image()
        isVertical = self.banding_direction.currentText() == "Vertical"
        if self.median_btn.isChecked():
            new_img = self.SirilRemoveBanding(img, 
                                        strength=self.strength_slider.value()/100,
                                        sigma=self.sigma_slider.value()/10,
                                        vertical=isVertical)
        if self.means_btn.isChecked():
            new_img = self.PixRemoveBanding(img, 
                                        strength=self.strength_slider.value()/100,
                                        sigma=self.sigma_slider.value()/10,
                                        passes=self.num_passes.value(),
                                        vertical=isVertical)
        if self.linear_btn.isChecked():
            new_img = self.LinearPatternSubtraction(img, 
                                        strength=self.strength_slider.value()/100,
                                        layer=self.num_wavelets.value(),
                                        sigma=self.sigma_slider.value()/10,
                                        vertical=isVertical)
        self.ctx.set_image(new_img, step_name="Remove Banding")
        self.apply_btn.setEnabled(True)
    
    def OnHelp(self):
        return

    def PixRemoveBanding(self, image, strength=0.5, sigma=3.0, passes=1, vertical=False):
        """
        Remove horizontal (or vertical) banding using PixInsight-style logic.

        Differences from the Siril/median approach:
          - Uses sigma-clipped MEAN per line (not median).
          - Supports multiple passes; each pass tightens rejection by
            reducing sigma by (sigma / passes) per step, converging on
            the banding pattern incrementally.

        Parameters
        ----------
        image    : np.ndarray  float image, shape (H, W) or (H, W, C), range 0-1
        strength : float       correction amount, 0.0 = none, 1.0 = full
        sigma    : float       starting sigma threshold for clipping (default 3.0)
        passes   : int         number of correction passes (default 1)
        vertical : bool        True → remove vertical banding instead of horizontal
        """
        result = image.astype(np.float32)
        squeeze = result.ndim == 2
        if squeeze:
            result = result[:, :, np.newaxis]

        h, w, num_channels = result.shape
        sigma_step = sigma / max(passes, 1)

        for ch in range(num_channels):
            for p in range(passes):
                pass_sigma = sigma - p * sigma_step
                pass_sigma = max(pass_sigma, 0.5)  # floor to avoid over-clipping

                plane = result[:, :, ch]
                if vertical:
                    plane = plane.T

                num_lines = plane.shape[0]
                line_means = np.empty(num_lines)

                for i in range(num_lines):
                    line_means[i] = sigma_clipped_mean(plane[i], pass_sigma)

                global_mean  = sigma_clipped_mean(line_means, pass_sigma)
                corrections  = (line_means - global_mean) * strength

                corrected = plane - corrections[:, np.newaxis]

                if vertical:
                    corrected = corrected.T

                result[:, :, ch] = corrected

        result = np.clip(result, 0.0, 1.0)
        return result[:, :, 0] if squeeze else result

    def LinearPatternSubtraction(self, image, strength=0.5, layer=2, sigma=3.0, vertical=False):
        """
        Wavelet-based linear pattern subtraction, analogous to PixInsight's
        LinearPatternSubtraction script.

        Unlike simple row-median removal (RemoveBanding / PixRemoveBanding),
        this works inside a specific à trous wavelet detail plane so it targets
        only the spatial frequency where the banding actually lives.  Large-scale
        sky gradients that happen to vary row-by-row are left untouched because
        they live in coarser layers that are never modified.

        Algorithm
        ---------
        1. Compute the à trous (starlet) detail plane at `layer`:
               detail = smooth(image, level=layer-1) - smooth(image, level=layer)
           Features at ~2^layer pixels wide are isolated in this plane.
        2. In the detail plane, compute the sigma-clipped median of each row
           (or column) → the 1-D banding profile at that scale.
        3. Subtract (profile * strength) from the *original* image.

        Parameters
        ----------
        image    : np.ndarray  float image, shape (H, W) or (H, W, C), range 0-1
        strength : float       correction amount, 0.0 = none, 1.0 = full
        layer    : int         wavelet scale to target; 1=finest (~2 px),
                               2=~4 px (default), 3=~8 px, etc.
        sigma    : float       sigma for row-profile clipping (default 3.0)
        vertical : bool        True → remove vertical banding instead of horizontal
        """
        result = image.astype(np.float32)
        squeeze = result.ndim == 2
        if squeeze:
            result = result[:, :, np.newaxis]

        h, w, num_channels = result.shape

        for ch in range(num_channels):
            plane = result[:, :, ch]

            # Build the detail layer at the target scale using à trous smoothing.
            # detail = c_{layer-1} - c_{layer}  (difference of two smoothed planes)
            c_coarse = atrous_smooth(plane, layer)
            c_fine   = atrous_smooth(plane, layer - 1) if layer > 0 else plane
            detail   = c_fine - c_coarse

            if vertical:
                detail = detail.T

            num_lines    = detail.shape[0]
            line_offsets = np.empty(num_lines, dtype=np.float32)
            for i in range(num_lines):
                line_offsets[i] = sigma_clipped_median(detail[i], sigma)

            global_offset = sigma_clipped_median(line_offsets, sigma)
            corrections   = (line_offsets - global_offset) * strength

            if vertical:
                result[:, :, ch] -= corrections[:, np.newaxis].T
            else:
                result[:, :, ch] -= corrections[:, np.newaxis]

        result = np.clip(result, 0.0, 1.0)
        return result[:, :, 0] if squeeze else result

    def SirilRemoveBanding(self, image, strength=0.5, sigma=3.0, vertical=False):
        """
        Remove horizontal (or vertical) banding noise from an astronomical image.

        Implements the same approach as Siril's Canon banding removal:
          1. For each row (or column if vertical=True), compute a sigma-clipped
             median to get a robust estimate of that line's background level.
          2. Compute the global sigma-clipped median across all line medians.
          3. The banding component for each line is its deviation from that
             global value.
          4. Subtract (banding_component * strength) from every pixel in the line.

        Parameters
        ----------
        image    : np.ndarray  float image, shape (H, W) or (H, W, C), range 0-1
        strength : float       correction amount, 0.0 = none, 1.0 = full
        sigma    : float       sigma threshold for clipping (default 3.0)
        vertical : bool        True → remove vertical banding instead of horizontal
        """
        result = image.astype(np.float32)
        squeeze = result.ndim == 2
        if squeeze:
            result = result[:, :, np.newaxis]

        h, w, num_channels = result.shape

        for ch in range(num_channels):
            plane = result[:, :, ch]
            if vertical:
                plane = plane.T          # treat columns as rows

            num_lines = plane.shape[0]
            line_medians = np.empty(num_lines)

            for i in range(num_lines):
                line_medians[i] = sigma_clipped_median(plane[i], sigma)

            global_median = sigma_clipped_median(line_medians, sigma)
            corrections   = (line_medians - global_median) * strength

            corrected = plane - corrections[:, np.newaxis]

            if vertical:
                corrected = corrected.T

            result[:, :, ch] = corrected

        result = np.clip(result, 0.0, 1.0)
        return result[:, :, 0] if squeeze else result


def atrous_smooth(plane, level):
    """
    2-D B3-spline à trous (with holes) smooth at the given dyadic scale level.

    The B3 kernel [1,4,6,4,1]/16 is applied separably along rows then columns.
    At level j the kernel taps are spaced 2^j pixels apart (zeros in between),
    giving a passband centred on features ~2^j pixels wide.
    Boundary handling uses nearest-pixel replication (clamp).
    """
    step = 1 << level          # 2^level
    k    = np.array([1/16, 4/16, 6/16, 4/16, 1/16], dtype=np.float32)
    offsets = np.array([-2, -1, 0, 1, 2]) * step

    h, w = plane.shape

    # --- along columns (axis=1) ---
    tmp = np.zeros_like(plane)
    cols = np.arange(w)
    for ki, off in zip(k, offsets):
        idx = np.clip(cols + off, 0, w - 1)
        tmp += ki * plane[:, idx]

    # --- along rows (axis=0) ---
    out = np.zeros_like(plane)
    rows = np.arange(h)
    for ki, off in zip(k, offsets):
        idx = np.clip(rows + off, 0, h - 1)
        out += ki * tmp[idx, :]

    return out

def sigma_clipped_mean(data, sigma=3.0, max_iter=10):
    """Return the sigma-clipped mean of a 1-D array."""
    d = data[np.isfinite(data)]
    for _ in range(max_iter):
        if d.size == 0:
            return 0.0
        mean = np.mean(d)
        std  = np.std(d)
        if std == 0.0:
            break
        keep = np.abs(d - mean) <= sigma * std
        if keep.sum() == d.size:
            break
        d = d[keep]
    return np.mean(d) if d.size > 0 else 0.0

def sigma_clipped_median(data, sigma=3.0, max_iter=10):
    """Return the sigma-clipped median of a 1-D array."""
    d = data[np.isfinite(data)]
    for _ in range(max_iter):
        if d.size == 0:
            return 0.0
        med = np.median(d)
        std = np.std(d)
        if std == 0.0:
            break
        keep = np.abs(d - med) <= sigma * std
        if keep.sum() == d.size:
            break
        d = d[keep]
    return np.median(d) if d.size > 0 else 0.0


def run(ctx):
    """
    SASpro entry point.
    """
    win = BandingReductionWindow(ctx, parent=ctx.app)
    win.setModal(False)
    win.setWindowModality(Qt.WindowModality.NonModal)
    win.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_star_reducer_window", win)
    except Exception:
        pass

    return win
