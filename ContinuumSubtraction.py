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
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
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
        self.CreateWidgets()

        # initialize some member variables
        self.r_file = ""
        self.g_file = ""
        self.b_file = ""
        self.ha_file = ""
        self.sii_file = ""
        self.oiii_file = ""
        self.emission_file = ""
        self.component_file = ""
        self.cs_file = ""

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

        self.r_line = QLineEdit()
        self.r_line.setReadOnly(True)
        row, btn = self.AddFileRow("R:", self.r_line, COMPONENT_LABEL_WIDTH)
        btn.clicked.connect(lambda: self.OnSelectFile("r_file", self.r_line))
        right_col.addLayout(row)

        self.g_line = QLineEdit()
        self.g_line.setReadOnly(True)
        row, btn = self.AddFileRow("G:", self.g_line, COMPONENT_LABEL_WIDTH)
        btn.clicked.connect(lambda: self.OnSelectFile("g_file", self.g_line))
        right_col.addLayout(row)

        self.b_line = QLineEdit()
        self.b_line.setReadOnly(True)
        row, btn = self.AddFileRow("B:", self.b_line, COMPONENT_LABEL_WIDTH)
        btn.clicked.connect(lambda: self.OnSelectFile("b_file", self.b_line))
        right_col.addLayout(row)

        self.emission_desc = QLabel("Emission Line Components", alignment=Qt.AlignmentFlag.AlignLeft)
        left_col.addWidget(self.emission_desc)

        self.ha_line = QLineEdit()
        self.ha_line.setReadOnly(True)
        row, btn = self.AddFileRow("Ha:", self.ha_line, EMISSION_LABEL_WIDTH)
        btn.clicked.connect(lambda: self.OnSelectFile("ha_file", self.ha_line))
        left_col.addLayout(row)

        self.sii_line = QLineEdit()
        self.sii_line.setReadOnly(True)
        row, btn = self.AddFileRow("SII:", self.sii_line, EMISSION_LABEL_WIDTH)
        btn.clicked.connect(lambda: self.OnSelectFile("sii_file", self.sii_line))
        left_col.addLayout(row)

        self.oiii_line = QLineEdit()
        self.oiii_line.setReadOnly(True)
        row, btn = self.AddFileRow("OIII:", self.oiii_line, EMISSION_LABEL_WIDTH)
        btn.clicked.connect(lambda: self.OnSelectFile("oiii_file", self.oiii_line))
        left_col.addLayout(row)

        comps_box.layout().addLayout(left_col, 1)
        comps_box.layout().addLayout(right_col, 1)
        layout.addWidget(comps_box)
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

    def AddFileRow(self, label_text, lineedit, label_width):
        """ Helper to create a file selection row """
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(label_width)
        row.addWidget(label)
        row.addWidget(lineedit, 1)
        btn = QPushButton("Select")
        row.addWidget(btn)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        return row, btn
    
    def OnEmissionChanged(self, text: str):
        """ Drop-down callback for selection change"""
        if text == "Ha":
            self.emission_file = self.ha_file
            self.component_file = self.r_file
            self.red_slider.setValue(100)
            self.green_slider.setValue(0)
            self.blu_slider.setValue(0)
        if text == "SII":
            self.emission_file = self.sii_file
            self.component_file = self.g_file
            self.red_slider.setValue(0)
            self.green_slider.setValue(100)
            self.blu_slider.setValue(0)
        if text == "OIII":
            self.emission_file = self.oiii_file
            self.component_file = self.b_file
            self.red_slider.setValue(0)
            self.green_slider.setValue(0)
            self.blu_slider.setValue(100)

    def OnSelectFile(self, file_attr: str, lineedit: QLineEdit):
        """ File selection button callback """
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "FITS files (*.fits *.fit *.fts *.fits.gz *.fit.gz *.fz *.fz2);;All files (*)")
        if path:
            lineedit.setText(os.path.basename(path))
            setattr(self, file_attr, path)
            self.OnEmissionChanged(self.emission_combo.currentText())

    def OnGenerate(self):
        """ Generate button callback. Generate the continuum subtracted image, and load into SASpro """
        c = self.c_slider.value() / 10000.0

        if not self.component_file or not self.emission_file:
            QMessageBox.warning(self, "Missing files", "Please select both Emission and Color component files.")
            return

        try:
            component_data = fits.getdata(self.component_file)
            emission_data = fits.getdata(self.emission_file)
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
        if not all([self.r_file, self.g_file, self.b_file]):
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
            r_data = fits.getdata(self.r_file).astype(np.float32)
            g_data = fits.getdata(self.g_file).astype(np.float32)
            b_data = fits.getdata(self.b_file).astype(np.float32)

            cs_median = np.median(cs_data)
            new_rdata = r_data + (cs_data - cs_median) * q * red_adjust
            new_gdata = g_data + (cs_data - cs_median) * q * green_adjust
            new_bdata = b_data + (cs_data - cs_median) * q * blue_adjust

            # Ensure output shape (3, height, width) as SASpro expects planes-first format
            combined_data = np.array([new_rdata, new_gdata, new_bdata], dtype=np.float32)

            # grab the fits header from one of the input files (R)
            with fits.open(self.r_file) as hdul:
                header = hdul[0].header

            c = self.c_slider.value() / 10000.0
            header.add_history(f"Continuum subtracted: emission={os.path.basename(self.emission_file)} c={c:.4f}")
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
        # compute the median of the emission file
        emission_data = fits.getdata(self.emission_file)
        c_median = np.median(emission_data)

        # load the narrowband and continuum data
        narrowband_data = fits.getdata(self.emission_file)
        continuum_data = fits.getdata(self.component_file)

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
