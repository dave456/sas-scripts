import os
import threading
import numpy as np
import cv2
from scipy.ndimage import median_filter

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QComboBox, QMessageBox, QGroupBox, QSlider, QRadioButton,
    QSpinBox, QFileDialog, QDoubleSpinBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from astropy.io import fits

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Blur Filters"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""   # optional

class BlurWindow(QDialog):
    finished_signal = pyqtSignal(object, str, str)  # result image, operation, error message

    def __init__(self, ctx, parent):
        """ Constructor for our UI class """
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Blur Filters")
        self.setFixedWidth(450)
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(1000)
        self.progress_timer.timeout.connect(self.OnProgressTick)
        self.finished_signal.connect(self.OnFinished)
        self.CreateWidgets()

    def CreateWidgets(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        algo_box = QGroupBox(" Settings ")
        algo_layout = QVBoxLayout()
        algo_box.setLayout(algo_layout)

        # blur algo selector
        combo_row = QHBoxLayout()
        combo_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.blur_combo = QComboBox()
        self.blur_combo.addItems(["Gaussian Blur", "Median Filter 3x3", "Bi-lateral Blur"])
        self.blur_combo.setFixedWidth(150)
        self.blur_combo.currentTextChanged.connect(self.OnAlgorithmChanged)
        combo_row.addWidget(self.blur_combo)
        algo_layout.addLayout(combo_row)

        algo_layout.addSpacing(10)

        # kernel size selector
        kernel_row = QHBoxLayout()
        kernel_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        kernel_label = QLabel(" Kernel Size:")
        kernel_label.setFixedWidth(80)
        kernel_row.addWidget(kernel_label)

        self.kernel_spin = QSpinBox()
        self.kernel_spin.setFixedWidth(80)
        self.kernel_spin.setRange(1, 99)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(1)
        self.kernel_spin.valueChanged.connect(self.OnKernelValueChanged)        
        kernel_row.addWidget(self.kernel_spin)

        self.kernel_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_slider.setRange(1, 99)
        self.kernel_slider.setValue(1)
        self.kernel_slider.setSingleStep(2)
        self.kernel_slider.setPageStep(10)
        self.kernel_slider.valueChanged.connect(self.OnKernelValueChanged)
        kernel_row.addWidget(self.kernel_slider)
        algo_layout.addLayout(kernel_row)

        # sigma selector (for Gaussian Blur)
        sigma_row = QHBoxLayout()
        sigma_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        sigma_label = QLabel(" Sigma:")
        sigma_label.setFixedWidth(80)
        sigma_row.addWidget(sigma_label)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setFixedWidth(80)
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(0.0)
        self.sigma_spin.valueChanged.connect(self.OnSigmaSpinChanged)
        sigma_row.addWidget(self.sigma_spin)

        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(0, 1000)
        self.sigma_slider.setValue(0)
        self.sigma_slider.setSingleStep(10)
        self.sigma_slider.setPageStep(50)
        self.sigma_slider.valueChanged.connect(self.OnSigmaSliderChanged)
        sigma_row.addWidget(self.sigma_slider)
        algo_layout.addLayout(sigma_row)

        # sigma color selector (for Bi-lateral Blur)
        sigma_color_row = QHBoxLayout()
        sigma_color_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        sigma_color_label = QLabel(" Sigma Color:")
        sigma_color_label.setFixedWidth(80)
        sigma_color_row.addWidget(sigma_color_label)

        self.sigma_color_spin = QDoubleSpinBox()
        self.sigma_color_spin.setFixedWidth(80)
        self.sigma_color_spin.setRange(0.001, 1.0)
        self.sigma_color_spin.setDecimals(3)
        self.sigma_color_spin.setSingleStep(0.01)
        self.sigma_color_spin.setValue(0.1)
        sigma_color_row.addWidget(self.sigma_color_spin)
        algo_layout.addLayout(sigma_color_row)

        self.OnAlgorithmChanged()

        layout.addWidget(algo_box)
        layout.addSpacing(10)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
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

    def OnAlgorithmChanged(self):
        algorithm = self.blur_combo.currentText()
        uses_kernel = algorithm in ("Gaussian Blur", "Bi-lateral Blur")
        self.sigma_spin.setEnabled(uses_kernel)
        self.sigma_slider.setEnabled(uses_kernel)
        self.kernel_spin.setEnabled(uses_kernel)
        self.kernel_slider.setEnabled(uses_kernel)
        self.sigma_color_spin.setEnabled(algorithm == "Bi-lateral Blur")

    def OnKernelValueChanged(self, value):
        """Kenel size should always be odd"""
        if value % 2 == 0:
            self.kernel_spin.blockSignals(True)
            self.kernel_spin.setValue(max(1, value - 1))
            self.kernel_spin.blockSignals(False)
        self.kernel_slider.blockSignals(True)
        self.kernel_slider.setValue(max(1, value))
        self.kernel_slider.blockSignals(False)

    def OnKernelSliderChanged(self, value):
        """Update kernel spinbox when slider changes"""
        #if value % 2 == 0:
        #    value -= 1
        self.kernel_spin.blockSignals(True)
        self.kernel_spin.setValue(max(1, value))
        self.kernel_spin.blockSignals(False)

    def OnSigmaSliderChanged(self, value):
        """Update sigma spinbox when slider changes"""
        sigma_value = value / 10.0
        self.sigma_spin.blockSignals(True)
        self.sigma_spin.setValue(sigma_value)
        self.sigma_spin.blockSignals(False)

    def OnSigmaSpinChanged(self, value):
        """Update sigma slider when spinbox changes"""
        slider_value = int(value * 10.0)
        self.sigma_slider.blockSignals(True)
        self.sigma_slider.setValue(slider_value)
        self.sigma_slider.blockSignals(False)

    def OnApply(self):
        self.apply_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        img = self.ctx.get_image()
        algorithm = self.blur_combo.currentText()
        kernel = self.kernel_spin.value()
        sigma = self.sigma_spin.value()
        sigma_color = self.sigma_color_spin.value()
        self.progress_timer.start()
        threading.Thread(target=self.Worker, args=(img, algorithm, kernel, sigma, sigma_color), daemon=True).start()

    def Worker(self, img, algorithm, kernel, sigma, sigma_color):
        try:
            if algorithm == "Gaussian Blur":
                new_img = self.GaussianBlur(img, kernel, sigma)
            elif algorithm == "Bi-lateral Blur":
                new_img = self.BilateralBlur(img, kernel, sigma, sigma_color)
            else:
                new_img = self.MedianFilter(img)
            self.finished_signal.emit(new_img, algorithm, "")
        except Exception as e:
            self.finished_signal.emit(None, algorithm, str(e))

    def OnProgressTick(self):
        # Fake progress: 5% per second, held at 95% until the operation completes.
        self.progress_bar.setValue(min(95, self.progress_bar.value() + 5))

    def OnFinished(self, new_img, operation, error):
        self.progress_timer.stop()
        if error:
            self.progress_bar.setValue(0)
            QMessageBox.critical(self, "Blur", error)
        else:
            self.ctx.set_image(new_img, operation)
            self.progress_bar.setValue(100)
        self.apply_btn.setEnabled(True)
    
    def OnHelp(self):
        return
    
    def GaussianBlur(self, image, kernel_size, sigma = 0.0):
        """Applies Gaussian Blur to the input image.
        Parameters:
            image (numpy.ndarray): Input image to be blurred.
            kernel_size (int): Size of the Gaussian kernel (must be odd and non-zero).
            sigma (float): Standard deviation for the Gaussian kernel. Default is 0.0, which means it will be calculated based on the kernel size."""
        k = max(1, int(kernel_size))
        if k % 2 == 0:
            k -= 1
        return cv2.GaussianBlur(image, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    
    def BilateralBlur(self, image, diameter, sigma_space, sigma_color):
        """Applies a bilateral filter, smoothing while preserving edges.
        Parameters:
            image (numpy.ndarray): Input image, 1 or 3 channels, values in the 0..1 range.
            diameter (int): Pixel neighborhood diameter.
            sigma_space (float): Spatial sigma. If 0, it is derived from the diameter.
            sigma_color (float): Intensity sigma; larger values blend more dissimilar pixels."""
        d = max(1, int(diameter))
        space = float(sigma_space) if sigma_space > 0 else d / 2.0
        # OpenCV only supports uint8 and float32 and drops a trailing singleton channel.
        result = cv2.bilateralFilter(image.astype(np.float32), d, float(sigma_color), space)
        return result.reshape(image.shape).astype(image.dtype)

    def MedianFilter(self, image):
        """Applies Median Filter to the input image."""
        # Kernel size is fixed at 3x3 for median filter, and we don't want to blur across color channels.
        # Kernel sizes larger than 3x3 take a long time to compute and don't provide much benefit for typical 
        # noise reduction.
        return median_filter(image, size=(3, 3) + (1,) * (image.ndim - 2)).astype(image.dtype)

def run(ctx):
    """
    SASpro entry point.
    """
    win = BlurWindow(ctx, parent=ctx.app)
    win.setModal(False)
    win.setWindowModality(Qt.WindowModality.NonModal)
    win.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_template_window", win)
    except Exception:
        pass

    return win
