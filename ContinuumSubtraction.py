# Narrowband Continuum Subtraction script
# SPDX-License-Identifier: GPL-3.0
# Author: Dave Lindner (c) 2025 lindner234 <AT> gmail
"""
This script provides continuum subtraction for narrowband images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QCheckBox, QSlider,
    QComboBox, QMainWindow
)
from PyQt6.QtCore import Qt, QTimer
from astropy.io import fits
from scipy.optimize import curve_fit
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Continuum Subtraction"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""   # optional

version = "v1.0.0"

class CSDialog(QDialog):
    def __init__(self, ctx, parent):
        """ Constructor for our UI class """
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Continuum Subtraction")
        self.setFixedWidth(660)
        self.combos = {}   # component key -> source combo box
        self.CreateWidgets()

    def CreateWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Compoenents group
        comps_box = QGroupBox(" Components")
        comps_frame = QHBoxLayout()
        comps_box.setLayout(comps_frame)
        comps_box.setContentsMargins(8, 23, 8, 13)
    
        left_col = QVBoxLayout()
        left_col.setSpacing(3)
        right_col = QVBoxLayout()
        right_col.setSpacing(3)

        # ui constants
        COMPONENT_LABEL_WIDTH = 10
        EMISSION_LABEL_WIDTH = 20

        self.cont_desc = QLabel("Color Components")
        right_col.addWidget(self.cont_desc, alignment=Qt.AlignmentFlag.AlignLeft)

        for key, text in (("r", "R:"), ("g", "G:"), ("b", "B:")):
            right_col.addLayout(self.AddSourceRow(key, text, COMPONENT_LABEL_WIDTH))

        self.emission_desc = QLabel("Emission Line Components", alignment=Qt.AlignmentFlag.AlignLeft)
        left_col.addWidget(self.emission_desc)

        for key, text in (("ha", "Ha:"), ("sii", "SII:"), ("oiii", "OIII:")):
            left_col.addLayout(self.AddSourceRow(key, text, EMISSION_LABEL_WIDTH))

        comps_box.layout().addLayout(left_col, 1)
        comps_box.layout().addLayout(right_col, 1)
        layout.addWidget(comps_box)

        refresh_btn = QPushButton("Refresh views")
        refresh_btn.clicked.connect(self.RefreshViews)
        layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.RefreshViews()
        layout.addSpacing(10)

        # CS generation group
        csgen_box = QGroupBox(" Continuum Subtraction Generation")
        csgen_frame = QVBoxLayout()
        csgen_box.setLayout(csgen_frame)
        csgen_box.setContentsMargins(8, 23, 8, 13)
        
        # drop-down to select which emission line to operate on
        self.emission_desc = QLabel("Emission Line Selection")
        csgen_box.layout().addWidget(self.emission_desc)

        self.emission_combo = QComboBox()
        self.emission_combo.addItems(["Ha", "SII", "OIII"])
        self.emission_combo.setCurrentIndex(0) # default to Ha
        self.emission_combo.setFixedWidth(70)
        self.emission_combo.currentTextChanged.connect(self.OnEmissionChanged)
        csgen_box.layout().addWidget(self.emission_combo)

        # load and estimate buttons
        btn_row = QHBoxLayout()
        estimate_btn = QPushButton("Estimate")
        estimate_btn.clicked.connect(self.OnEstimate)
        self.plot_check_box = QCheckBox("Plot Solution")
        btn_row.addWidget(estimate_btn)
        btn_row.addWidget(self.plot_check_box)
        btn_row.addStretch()
        csgen_box.layout().addLayout(btn_row)
        csgen_box.layout().addSpacing(10)

        # c constant continuum slider
        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("c:"))
        self.c_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_slider.setRange(0, 10000)
        self.c_slider.setValue(2000)
        c_row.addWidget(self.c_slider)
        self.c_value_label = QLabel(f"{self.c_slider.value() / 10000:.4f}")
        c_row.addWidget(self.c_value_label)
        self.c_slider.valueChanged.connect(lambda v: self.c_value_label.setText(f"{v / 10000:.4f}"))
        csgen_box.layout().addLayout(c_row)
        csgen_box.layout().addSpacing(5)
        
        # generate button
        gen_btn_row = QHBoxLayout()
        gen_btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gen_btn = QPushButton("Generate")
        gen_btn.setToolTip("Generate the continuum subtracted image based on the selected emission line and scaling factor c.")
        gen_btn.clicked.connect(self.OnGenerate)
        gen_btn.setFixedWidth(80)
        gen_help_btn = QPushButton("Help")
        gen_help_btn.clicked.connect(self.OnGenHelp)
        gen_help_btn.setFixedWidth(80)
        gen_btn_row.addWidget(gen_btn)
        gen_btn_row.addSpacing(40)
        gen_btn_row.addWidget(gen_help_btn)

        csgen_box.layout().addLayout(gen_btn_row)
        layout.addWidget(csgen_box)
        layout.addSpacing(5)

        # Blending group
        blend_box = QGroupBox(" Blending Options")
        blend_box_frame = QVBoxLayout()
        blend_box.setLayout(blend_box_frame)
        blend_box.setContentsMargins(8, 23, 8, 13)

        # q strength slider (determines Ha contribution to final image)
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("Strength (q):"))
        self.q_slider = QSlider(Qt.Orientation.Horizontal)
        self.q_slider.setRange(0, 1200)
        self.q_slider.setValue(200)
        q_row.addWidget(self.q_slider)
        self.q_value_label = QLabel(f"{self.q_slider.value() / 100:.2f}")
        q_row.addWidget(self.q_value_label)
        self.q_slider.valueChanged.connect(lambda v: self.q_value_label.setText(f"{v / 100:.2f}"))
        blend_box.layout().addLayout(q_row)

        blend_box.layout().addSpacing(6)  # fudge for spacing

        # ui constants
        COLOR_LABEL_WIDTH = 35
        VALUE_LABEL_WIDTH = 40

        # optional red channel blending slider
        red_slider_row = QHBoxLayout()
        red_label = QLabel("Red:")
        red_label.setFixedWidth(COLOR_LABEL_WIDTH)
        red_slider_row.addWidget(red_label)
        self.red_slider = QSlider(Qt.Orientation.Horizontal)
        self.red_slider.setRange(0, 100)
        self.red_slider.setValue(100)
        red_slider_row.addWidget(self.red_slider)
        self.red_value_label = QLabel(f"{self.red_slider.value()}%")
        self.red_value_label.setFixedWidth(VALUE_LABEL_WIDTH)
        self.red_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        red_slider_row.addWidget(self.red_value_label)
        self.red_slider.valueChanged.connect(lambda v: self.red_value_label.setText(f"{v}%"))
        blend_box.layout().addLayout(red_slider_row)

       # optional green channel blending slider
        green_slider_row = QHBoxLayout()
        green_label = QLabel("Green:")
        green_label.setFixedWidth(COLOR_LABEL_WIDTH)
        green_slider_row.addWidget(green_label)
        self.green_slider = QSlider(Qt.Orientation.Horizontal)
        self.green_slider.setRange(0, 100)
        self.green_slider.setValue(0)
        green_slider_row.addWidget(self.green_slider)
        self.green_value_label = QLabel(f"{self.green_slider.value()}%")
        self.green_value_label.setFixedWidth(VALUE_LABEL_WIDTH)
        self.green_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        green_slider_row.addWidget(self.green_value_label)
        self.green_slider.valueChanged.connect(lambda v: self.green_value_label.setText(f"{v}%"))
        blend_box.layout().addLayout(green_slider_row)

        # optional blue channel blending slider
        blu_slider_row = QHBoxLayout()
        blu_label = QLabel("Blue:")
        blu_label.setFixedWidth(COLOR_LABEL_WIDTH)
        blu_slider_row.addWidget(blu_label)
        self.blu_slider = QSlider(Qt.Orientation.Horizontal)
        self.blu_slider.setRange(0, 100)
        self.blu_slider.setValue(0)
        blu_slider_row.addWidget(self.blu_slider)
        self.blu_value_label = QLabel(f"{self.blu_slider.value()}%")
        self.blu_value_label.setFixedWidth(VALUE_LABEL_WIDTH)
        self.blu_value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        blu_slider_row.addWidget(self.blu_value_label)
        self.blu_slider.valueChanged.connect(lambda v: self.blu_value_label.setText(f"{v}%"))
        blend_box.layout().addLayout(blu_slider_row)

        blend_box.layout().addSpacing(15)  # fudge for spacing

        # Blend button
        blend_btn_row = QHBoxLayout()
        blend_btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        blend_btn = QPushButton("Blend")
        blend_btn.clicked.connect(self.OnBlend)
        blend_btn.setFixedWidth(80)
        help_blend_btn = QPushButton("Help")
        help_blend_btn.clicked.connect(self.OnBlendHelp)
        help_blend_btn.setFixedWidth(80)
        blend_btn_row.addWidget(blend_btn)
        blend_btn_row.addSpacing(40)
        blend_btn_row.addWidget(help_blend_btn)
        blend_box.layout().addLayout(blend_btn_row)

        layout.addWidget(blend_box)

        self.status_text = QLabel("")
        layout.addWidget(self.status_text)

    def OnGenHelp(self):
        QMessageBox.information(self, "Help - Continuum Subtraction Generation", 
            "Select the emission line you want to subtract (Ha, SII, OIII) from the drop-down.\n\n"
            "Adjust the scaling factor 'c' using the slider. This determines how much of the continuum "
            "component is subtracted from the emission image. You can also click 'Estimate' to have the "
            "script compute an optimal value for 'c'.\n\n"
            "Once you are satisfied with the selected emission line and scaling factor, click 'Generate' "
            "to create the continuum subtracted image and it will automatically be loaded into SASpro."
        )

    def OnBlendHelp(self):
        QMessageBox.information(self, "Help - Blending Options", 
            "The blending options allow you to create a blended RGB image using the continuum subtracted image.\n\n"
            "Adjust the 'Strength (q)' slider to control how much of the continuum subtracted image is added back into the RGB channels. "
            "Higher values will make the emission features more prominent in the final blended image.\n\n"
            "You can also adjust the Red, Green, and Blue sliders to control how much of the continuum subtracted image is added to each respective color channel. "
            "This allows you to customize the color balance of the blended image.\n\n"
        )

    def AddSourceRow(self, key, label_text, label_width):
        """ Helper to create a row with a source combo (open views or a file) """
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(label_width)
        row.addWidget(label)
        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        self.combos[key] = combo
        row.addWidget(combo, 1)
        btn = QPushButton("File...")
        btn.clicked.connect(lambda _checked=False, k=key: self.OnSelectFile(k))
        row.addWidget(btn)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        return row

    def RefreshViews(self):
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

    def LoadSource(self, key):
        """ Returns (pixel data, FITS header or None) from the selected view or file """
        kind, value = self.combos[key].currentData()
        if kind == "file":
            with fits.open(value) as hdul:
                return hdul[0].data, hdul[0].header
        img = np.asarray(self.ctx.get_image_for(value))
        return (img.mean(axis=2) if img.ndim == 3 else img), None

    def EmissionKeys(self):
        """ Returns (emission key, matching color component key) for the selected emission line """
        return {"Ha": ("ha", "r"), "SII": ("sii", "g"), "OIII": ("oiii", "b")}[self.emission_combo.currentText()]

    def HasSources(self, *keys):
        return all(self.combos[k].currentData() for k in keys)
    
    def OnEmissionChanged(self, text: str):
        """ Drop-down callback for selection change"""
        if text == "Ha":
            self.red_slider.setValue(100)
            self.green_slider.setValue(0)
            self.blu_slider.setValue(0)
        if text == "SII":
            self.red_slider.setValue(0)
            self.green_slider.setValue(100)
            self.blu_slider.setValue(0)
        if text == "OIII":
            self.red_slider.setValue(0)
            self.green_slider.setValue(0)
            self.blu_slider.setValue(100)

    def OnSelectFile(self, key):
        """ File selection button callback """
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "FITS files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz *.fz2);;All files (*)")
        if path:
            combo = self.combos[key]
            data = ("file", path)
            index = combo.findData(data)
            if index < 0:
                combo.addItem(os.path.basename(path), data)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

    def OnGenerate(self):
        """ Generate button callback. Generate the continuum subtracted image, and load into SASpro """
        c = self.c_slider.value() / 10000.0
        emission_key, component_key = self.EmissionKeys()

        if not self.HasSources(emission_key, component_key):
            QMessageBox.warning(self, "Missing sources", "Please select both Emission and Color component views or files.")
            return

        try:
            component_data, _ = self.LoadSource(component_key)
            emission_data, _ = self.LoadSource(emission_key)
            cs_data = emission_data - c * component_data
 
            # Load into SASpro
            doc = self.ctx.get_document("CS-generated")
            if doc:
                self.ctx.set_image_for("CS-generated", cs_data, step_name='Update CS')
            else:
                self.ctx.open_new_document(cs_data, metadata=None, name="CS-generated")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating continuum subtracted image:\n{e}")

    def OnBlend(self):
        if not self.HasSources("r", "g", "b"):
            QMessageBox.warning(self, "Missing components", "Please select R, G, B and ensure the continuum subtracted image is generated.")
            return

        try:
            q = self.q_slider.value() / 100.0
            red_adjust = self.red_slider.value() / 100.0
            blue_adjust = self.blu_slider.value() / 100.0
            green_adjust = self.green_slider.value() / 100.0

            # load our generated CS image
            cs_doc = self.ctx.get_document("CS-generated")
            if cs_doc is None:
                QMessageBox.warning(self, "Missing CS Image", "Please generate the continuum subtracted image before blending.")
                return

            cs_data = cs_doc.image.astype(np.float32)
            r_data, r_header = self.LoadSource("r")
            r_data = r_data.astype(np.float32)
            g_data = self.LoadSource("g")[0].astype(np.float32)
            b_data = self.LoadSource("b")[0].astype(np.float32)

            cs_median = np.median(cs_data)
            new_rdata = r_data + (cs_data - cs_median) * q * red_adjust
            new_gdata = g_data + (cs_data - cs_median) * q * green_adjust
            new_bdata = b_data + (cs_data - cs_median) * q * blue_adjust

            # Ensure output shape (3, height, width) as SASpro expects planes-first format
            combined_data = np.array([new_rdata, new_gdata, new_bdata], dtype=np.float32)

            # use the R file's fits header; views have none
            header = r_header if r_header is not None else fits.Header()

            c = self.c_slider.value() / 10000.0
            emission_name = self.combos[self.EmissionKeys()[0]].currentText()
            header.add_history(f"Continuum subtracted: emission={emission_name} c={c:.4f}")
            combined_data = np.transpose(combined_data, (1, 2, 0))

            # update or open our blended view in SASpro
            blended_doc = self.ctx.get_document("CS-blended")
            if blended_doc:
                self.ctx.set_image_for("CS-blended", combined_data, step_name='Update Blended')
            else:
                self.ctx.open_new_document(combined_data, metadata=header, name="CS-blended")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error during blending:\n{e}")

    def OnEstimate(self):
        emission_key, component_key = self.EmissionKeys()
        if not self.HasSources(emission_key, component_key):
            QMessageBox.warning(self, "Missing sources", "Please select both Emission and Color component views or files.")
            return

        # load the narrowband and continuum data
        narrowband_data, _ = self.LoadSource(emission_key)
        continuum_data, _ = self.LoadSource(component_key)
        c_median = np.median(narrowband_data)

        # verify shapes match, e.g. same dimensions, mono images, etc.
        if continuum_data.shape != narrowband_data.shape:
            QMessageBox.critical(self, "Mismatched Images", "Continuum and narrowband image sizes and types must match.")
            return
        
        c = self.compute_c(
            narrowband_data,
            continuum_data,
            c_median,
            self.plot_check_box.isChecked()
        )

        self.ctx.log(f"Estimated continuum scaling factor c: {c:.4f}")
        self.c_slider.setValue(int(round(c * 10000)))
        self.status_text.setText("")
        self.OnGenerate()

    def compute_c(self, narrowband_image, continuum_image, c_median, plot_optimization):
        """ Compute the optimal continuum scaling factor c """
        approx_min = find_min(narrowband_image, continuum_image, c_median)
        max_val = approx_min + 1.0
        min_val = approx_min - 1.0

        scale_factors = np.linspace(min_val, max_val, 40)
        aad_values = []

        self.status_text.setText("Optimizing continuum subtraction...")
        for i, sf in enumerate(scale_factors):
            value = aad(narrowband_image - (continuum_image - c_median) * sf)
            aad_values.append(value)

        def smooth_v(x, A, s0, eps, B):
            return A * np.sqrt((x - s0)**2 + eps**2) + B

        B0 = np.min(aad_values)
        s0_0 = scale_factors[np.argmin(aad_values)]
        slope_est = (aad_values[-1] - aad_values[0]) / (scale_factors[-1] - scale_factors[0])
        A0 = slope_est
        eps0 = 0.01
        p0 = [A0, s0_0, eps0, B0]
        lb = [-1.0, 0.00, 0.0, 0.00]
        ub = [np.inf, 2*max_val, np.inf, np.inf]

        popt, _ = curve_fit(smooth_v, scale_factors, aad_values, p0=p0, bounds=(lb, ub))
        A_opt, s0_opt, eps_opt, B_opt = popt
        c = float(np.clip(s0_opt, 0, 1))

        if plot_optimization and self is not None:
            def show_plot():
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(scale_factors, aad_values, color='C0', alpha=0.6, label='AAD values')
                fx = np.linspace(min_val, max_val, 500)
                fy = smooth_v(fx, *popt)
                ax.plot(fx, fy, 'C3-', label="Smooth-V fit")
                min_aad = smooth_v(c, *popt)
                ax.plot([c], [min_aad], 'go', ms=10, label=f'Optimal scale = {c:.4f}')
                ax.axvline(c, color='green', ls='--', alpha=0.5)
                ax.set_title('Optimization for Continuum Subtraction')
                ax.set_xlabel('Scale Factor')
                ax.set_ylabel('AAD')
                ax.grid(alpha=0.3)
                ax.legend(loc='best')

                plot_window = QMainWindow(self)
                plot_window.setWindowTitle("Continuum Subtraction Optimization")
                canvas = FigureCanvasQTAgg(fig)
                central = QWidget()
                layout = QVBoxLayout(central)
                layout.addWidget(canvas)
                plot_window.setCentralWidget(central)
                plot_window.resize(800, 600)
                plot_window.show()

            QTimer.singleShot(0, show_plot)

        return c

def aad(data):
    mean = np.mean(data)
    return np.mean(np.abs(data - mean))

def find_min(nb, co, c_median):
    scale_factors = np.linspace(-1, 5, 12)
    aad_values = []

    for i, sf in enumerate(scale_factors):
        value = aad(nb - (co - c_median) * sf)
        aad_values.append(value)

    min_val = scale_factors[np.argmin(aad_values)]
    return min_val

def run(ctx):
    """
    SASpro entry point.
    """
    win = CSDialog(ctx, parent=ctx.app)
    win.setModal(False)
    win.setWindowModality(Qt.WindowModality.NonModal)
    win.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_continuum_subtraction_window", win)
    except Exception:
        pass

    return win
