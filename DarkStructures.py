"""
JR Dark Structure Enhance for Seti Astro Suite Pro
==================================================

Ported from the original Siril script by Jair Rillo Junior.

Enhances dust lanes and dark nebular structure with a wavelet-based
shadow mask and a soft shadow-deepening curve.

Best used on stretched, preferably starless images.

Inspired by the classic DarkStructureEnhance approach created by
Carlos Sonnenstein and implemented for PixInsight by Oriol Lehmkuhl
(PTeam, 2009). This tool uses an independent control model,
Rec.709 luminance, soft mask shaping, sky protection, and a
shadow-power deepen — not a 1:1 port of the original UI or pipeline.

Author: Jair Rillo Junior
Contact: jairrillo@gmail.com
Instagram: https://www.instagram.com/jairrillojunior/
YouTube: https://www.youtube.com/@jair_rillo
(c) 2026 Jair Rillo Junior
SPDX-License-Identifier: GPL-3.0-or-later

Credits: algorithm lineage — Carlos Sonnenstein & Oriol Lehmkuhl (PTeam).
"""

# Version History
# 1.0.0  Siril-native UI + improved algorithm (inspired by PTeam DSE lineage)
# 1.1.0  Starless mode (default), Help documentation, public naming
# SASpro port: dialog driven by the SASpro script context, no CLI mode

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve as nd_convolve
from scipy.ndimage import gaussian_filter

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# =========================
# SASpro Script Metadata
# =========================
SCRIPT_NAME     = "Dark Structures"
SCRIPT_GROUP    = "Tools"
SCRIPT_SHORTCUT = ""   # optional

VERSION = "1.0.1"
TITLE = "JR Dark Structure Enhance"
MASK_DOC_NAME = "JR_DSE_mask"

# User-facing defaults (not PI Amount/Layers/Iterations)
DEFAULT_STRENGTH = 0.45
DEFAULT_SCALE = 1          # 0=fine, 1=medium, 2=coarse
DEFAULT_STAR_GUARD = 1     # 0=low, 1=normal, 2=high
DEFAULT_MASK_SOFTNESS = 0.50
DEFAULT_SKY_PROTECT = 0.35
DEFAULT_STARLESS = True    # typical workflow: run on starless
DEFAULT_PRESET = 1         # Balanced

HELP_TEXT = """
<html><body style="font-size:11pt;">
<p><b>JR Dark Structure Enhance</b> deepens dust lanes and dark nebular structure.
Best results are usually on a <b>starless</b>, stretched image.</p>

<p><b>Preset</b><br>
Quick starting points: Subtle, Balanced, or Dramatic.
Tweaking any slider afterwards is fine — presets are just shortcuts.</p>

<p><b>Strength</b><br>
How strongly dark structures are deepened.
Low = gentle; high = more contrast in the lanes.</p>

<p><b>Structure size</b><br>
• <i>Fine dust</i> — smaller / thinner features<br>
• <i>Balanced</i> — general purpose<br>
• <i>Large lanes</i> — broader galactic dust structures</p>

<p><b>Starless image</b> (recommended)<br>
Enable when stars were already removed (StarNet, StarXTerminator, etc.).
The script then skips star-guard filtering so the mask follows nebula
and dust more cleanly. Disable only if the image still has stars.</p>

<p><b>Star protection</b> (with-stars mode only)<br>
How aggressively fine bright structures (stars) are kept out of
the shadow mask. Higher = safer for stars, slightly softer mask.</p>

<p><b>Mask softness</b><br>
Soft = natural, feathered transitions.
Hard = tighter selection that digs into fine dust.</p>

<p><b>Sky protect</b><br>
Limits crushing of near-black background sky.
Raise if the background looks too dark after Apply.</p>

<p><b>Create shadow mask view</b><br>
Opens (or updates) a <code>JR_DSE_mask</code> view containing the shadow mask
(useful for inspection or reuse as a mask).</p>

<p style="color:#666;"><i>Inspired by work of Carlos Sonnenstein &amp;
Oriol Lehmkuhl (PTeam). Independent implementation.</i></p>
</body></html>
"""


# ---------------------------------------------------------------------------
# Wavelets (à trous, B3 spline)
# ---------------------------------------------------------------------------

_B3_1D = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0


def _build_spaced_kernel(kernel: np.ndarray, scale_idx: int) -> np.ndarray:
    if scale_idx == 0:
        return kernel.astype(np.float32, copy=False)
    step = 2 ** scale_idx
    spaced_len = len(kernel) + (len(kernel) - 1) * (step - 1)
    spaced = np.zeros(spaced_len, dtype=np.float32)
    spaced[0::step] = kernel
    return spaced


def _conv_sep_reflect(image2d: np.ndarray, k1d: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        return nd_convolve(image2d, k1d.reshape(1, -1), mode="reflect")
    return nd_convolve(image2d, k1d.reshape(-1, 1), mode="reflect")


def atrous_decompose(img2d: np.ndarray, n_scales: int) -> list[np.ndarray]:
    current = img2d.astype(np.float32, copy=True)
    scales: list[np.ndarray] = []
    for i in range(n_scales):
        k = _build_spaced_kernel(_B3_1D, i)
        tmp = _conv_sep_reflect(current, k, axis=1)
        smooth = _conv_sep_reflect(tmp, k, axis=0)
        scales.append(current - smooth)
        current = smooth
    scales.append(current)
    return scales


def atrous_detail_sum(scales: list[np.ndarray], n_detail: int) -> np.ndarray:
    """Sum the first n_detail wavelet planes (stars / fine structure)."""
    out = np.zeros_like(scales[0], dtype=np.float32)
    for i in range(min(n_detail, len(scales) - 1)):
        out += scales[i]
    return out


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_chw(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    """SASpro images are HWC (or HW for mono)."""
    img = np.asarray(data, dtype=np.float32)
    if img.ndim == 2:
        return img[np.newaxis, ...], True
    if img.ndim == 3 and img.shape[2] in (1, 3):
        chw = np.transpose(img, (2, 0, 1))
        return chw, chw.shape[0] == 1
    raise ValueError(f"Unsupported image shape: {img.shape}")


def from_chw(chw: np.ndarray, was_mono: bool) -> np.ndarray:
    out = np.clip(chw, 0.0, 1.0).astype(np.float32, copy=False)
    return out[0] if was_mono else np.ascontiguousarray(np.transpose(out, (1, 2, 0)))


def luminance_rec709(chw: np.ndarray) -> np.ndarray:
    """Rec.709 luminance (not equal-weight)."""
    if chw.shape[0] == 1:
        return chw[0].astype(np.float32, copy=True)
    return (
        0.2126 * chw[0] + 0.7152 * chw[1] + 0.0722 * chw[2]
    ).astype(np.float32)


def rescale_01(img: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(img, 0.5))
    hi = float(np.percentile(img, 99.5))
    if hi <= lo:
        lo = float(np.min(img))
        hi = float(np.max(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Control mapping (user language → internal)
# ---------------------------------------------------------------------------

@dataclass
class UmbraParams:
    strength: float = DEFAULT_STRENGTH       # 0..1
    scale: int = DEFAULT_SCALE               # 0 fine / 1 medium / 2 coarse
    star_guard: int = DEFAULT_STAR_GUARD     # 0 low / 1 normal / 2 high
    mask_softness: float = DEFAULT_MASK_SOFTNESS
    sky_protect: float = DEFAULT_SKY_PROTECT
    starless: bool = DEFAULT_STARLESS


def _structure_depth(scale: int) -> int:
    """How many wavelet layers to peel before reading the residual."""
    return {0: 6, 1: 8, 2: 10}.get(int(scale), 8)


def _star_guard_depth(star_guard: int) -> int:
    """Fine scales stripped before residual extraction (star protection)."""
    return {0: 3, 1: 5, 2: 7}.get(int(star_guard), 5)


# ---------------------------------------------------------------------------
# Core algorithm — inspired lineage, independent pipeline
# ---------------------------------------------------------------------------

ProgressCB = Optional[Callable[[str, int], None]]


def build_umbra_mask(chw: np.ndarray, params: UmbraParams, progress_cb: ProgressCB = None) -> np.ndarray:
    """
    Build a soft dark-structure mask.

    Idea (shared lineage): dark lanes sit below the large-scale residual.
    Differences vs classic DSE:
      - Rec.709 luminance
      - adjustable star-guard depth
      - percentile rescale + Gaussian feather
      - power softness + sky gate
    """
    if progress_cb:
        progress_cb("Analyzing luminance...", 8)

    luma = luminance_rec709(chw)
    depth = _structure_depth(params.scale)

    if params.starless:
        # Starless workflow: no star-guard strip — work on luma directly
        if progress_cb:
            progress_cb("Starless mode — building residual...", 28)
        cleaned = luma
    else:
        guard = _star_guard_depth(params.star_guard)
        if progress_cb:
            progress_cb("Separating stars / fine structure...", 22)
        # Strip stars so they don't pollute the shadow mask
        scales_guard = atrous_decompose(luma, max(guard, 1))
        fine = atrous_detail_sum(scales_guard, guard)
        cleaned = np.clip(luma - fine, 0.0, 1.0)

    if progress_cb:
        progress_cb("Mapping large-scale residual...", 48)

    scales = atrous_decompose(cleaned, depth)
    residual = scales[-1]

    # Positive where the image is darker than its large-scale envelope
    raw = np.clip(residual - luma, 0.0, None)
    mask = rescale_01(raw)

    if progress_cb:
        progress_cb("Shaping mask...", 70)

    # Softness: higher → smoother / gentler mask falloff
    soft = float(np.clip(params.mask_softness, 0.0, 1.0))
    sigma = 0.6 + 3.4 * soft
    mask = gaussian_filter(mask, sigma=sigma)
    # Power curve: soft=0 → contrasty (gamma~0.7), soft=1 → airy (gamma~1.8)
    gamma = 0.70 + 1.10 * soft
    mask = np.power(np.clip(mask, 0.0, 1.0), gamma).astype(np.float32)

    # Sky protection: reduce effect on near-black background
    sky = float(np.clip(params.sky_protect, 0.0, 1.0))
    if sky > 1e-3:
        # Gate opens as luminance rises above a sky floor
        floor = 0.02 + 0.10 * sky
        gate = smoothstep(floor * 0.35, floor + 0.08 * sky, luma)
        # Also gently attenuate empty sky (very low luma)
        mask = mask * (0.25 + 0.75 * gate)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def deepen_shadows(chw: np.ndarray, mask: np.ndarray, strength: float,
                   progress_cb: ProgressCB = None) -> np.ndarray:
    """
    Soft shadow deepen through the mask.

    Uses a luminance-aware power curve (not PI midtones HT / iterations):
      darkened = luma ** (1 + k*strength)
    then remaps RGB by the luma ratio so colour is preserved.
    """
    if progress_cb:
        progress_cb("Deepening shadows...", 88)

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1e-4:
        return chw.astype(np.float32, copy=True)

    luma = luminance_rec709(chw)
    # Power > 1 darkens mid-shadows while anchoring 0 and 1
    exponent = 1.0 + 1.35 * strength
    luma_safe = np.clip(luma, 1e-6, 1.0)
    darkened = np.power(luma_safe, exponent).astype(np.float32)

    # Blend amount follows mask * strength (mask already shaped)
    blend = (mask * strength).astype(np.float32)
    new_luma = luma * (1.0 - blend) + darkened * blend

    # Preserve chromaticity: scale RGB by luma ratio
    ratio = np.ones_like(luma, dtype=np.float32)
    nz = luma > 1e-6
    ratio[nz] = new_luma[nz] / luma[nz]

    out = chw.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        out[c] = np.clip(out[c] * ratio, 0.0, 1.0)

    return out.astype(np.float32)


def umbra_enhance(data: np.ndarray, params: UmbraParams,
                  progress_cb: ProgressCB = None) -> Tuple[np.ndarray, np.ndarray, bool]:
    chw, was_mono = to_chw(data)
    chw = np.clip(chw, 0.0, 1.0).astype(np.float32, copy=False)

    mask = build_umbra_mask(chw, params, progress_cb=progress_cb)
    out = deepen_shadows(chw, mask, params.strength, progress_cb=progress_cb)

    if progress_cb:
        progress_cb("Done", 100)

    return from_chw(out, was_mono), mask, was_mono


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class UmbraWorker(QThread):
    progress_update = pyqtSignal(str, int)
    finished_ok = pyqtSignal(object, object)   # out, mask
    finished_err = pyqtSignal(str)

    def __init__(self, data: np.ndarray, params: UmbraParams):
        super().__init__()
        self.data = data
        self.params = params

    def run(self):
        try:
            def cb(msg, pct):
                self.progress_update.emit(msg, pct)

            out, mask, _ = umbra_enhance(self.data, self.params, progress_cb=cb)
            self.finished_ok.emit(out, mask)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_err.emit(str(e))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

SCALE_LABELS = ("Fine dust", "Balanced", "Large lanes")
GUARD_LABELS = ("Low", "Normal", "High")
PRESET_LABELS = ("Subtle", "Balanced", "Dramatic")


class UmbraDialog(QDialog):
    def __init__(self, ctx, parent):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"{TITLE} v{VERSION}")
        self.setMinimumWidth(400)
        self.setMaximumWidth(460)

        self._worker: Optional[UmbraWorker] = None
        self._last_mask: Optional[np.ndarray] = None

        self._build_ui()
        self._apply_preset(DEFAULT_PRESET)

    # --- UI construction ---------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel(TITLE)
        tf = QFont()
        tf.setPointSize(13)
        tf.setBold(True)
        title.setFont(tf)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("Deepen dust lanes and dark nebular structure")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888;")
        root.addWidget(subtitle)

        # Presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        for name in PRESET_LABELS:
            self.combo_preset.addItem(name)
        self.combo_preset.setCurrentIndex(DEFAULT_PRESET)
        self.combo_preset.currentIndexChanged.connect(self._on_preset)
        preset_row.addWidget(self.combo_preset, 1)
        root.addLayout(preset_row)

        # Main controls
        main = QGroupBox("Enhancement")
        main_l = QVBoxLayout(main)
        main_l.setSpacing(8)

        self.chk_starless = QCheckBox("Starless image (recommended)")
        self.chk_starless.setChecked(DEFAULT_STARLESS)
        self.chk_starless.setToolTip(
            "Enable when stars were already removed. "
            "Skips star-guard filtering for a cleaner dust mask."
        )
        self.chk_starless.toggled.connect(self._on_starless_toggled)
        main_l.addWidget(self.chk_starless)

        self.slider_strength, self.lbl_strength = self._add_slider(
            main_l, "Strength", 0, 100, int(DEFAULT_STRENGTH * 100),
            "How strongly dark structures are deepened."
        )

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Structure size:"))
        self.combo_scale = QComboBox()
        for name in SCALE_LABELS:
            self.combo_scale.addItem(name)
        self.combo_scale.setCurrentIndex(DEFAULT_SCALE)
        self.combo_scale.setToolTip(
            "Fine = smaller dust; Coarse = broader dark lanes."
        )
        self.combo_scale.currentIndexChanged.connect(self._mark_custom)
        scale_row.addWidget(self.combo_scale, 1)
        main_l.addLayout(scale_row)

        root.addWidget(main)

        # Refinement
        refine = QGroupBox("Refinement")
        refine_l = QVBoxLayout(refine)
        refine_l.setSpacing(8)

        self.guard_row_widget = QWidget()
        guard_row = QHBoxLayout(self.guard_row_widget)
        guard_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_guard = QLabel("Star protection:")
        guard_row.addWidget(self.lbl_guard)
        self.combo_guard = QComboBox()
        for name in GUARD_LABELS:
            self.combo_guard.addItem(name)
        self.combo_guard.setCurrentIndex(DEFAULT_STAR_GUARD)
        self.combo_guard.setToolTip(
            "Only used when the image still has stars. "
            "Higher = keep stars out of the shadow mask more aggressively."
        )
        self.combo_guard.currentIndexChanged.connect(self._mark_custom)
        guard_row.addWidget(self.combo_guard, 1)
        refine_l.addWidget(self.guard_row_widget)

        self.slider_soft, self.lbl_soft = self._add_slider(
            refine_l, "Mask softness", 0, 100, int(DEFAULT_MASK_SOFTNESS * 100),
            "Softer masks look more natural; harder masks dig into fine dust."
        )
        self.slider_sky, self.lbl_sky = self._add_slider(
            refine_l, "Sky protect", 0, 100, int(DEFAULT_SKY_PROTECT * 100),
            "Limits crushing of near-black background sky."
        )

        self.chk_save_mask = QCheckBox("Create shadow mask view")
        self.chk_save_mask.setToolTip(
            f"Opens or updates a '{MASK_DOC_NAME}' view with the shadow mask."
        )
        refine_l.addWidget(self.chk_save_mask)

        root.addWidget(refine)

        # Progress
        self.lbl_status = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.lbl_status)
        root.addWidget(self.progress)

        # Buttons
        btns = QHBoxLayout()
        self.btn_help = QPushButton("Help")
        self.btn_help.clicked.connect(self._show_help)
        btns.addWidget(self.btn_help)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(lambda: self._apply_preset(DEFAULT_PRESET))
        btns.addWidget(self.btn_reset)
        btns.addStretch()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_close)
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._apply)
        btns.addWidget(self.btn_apply)
        root.addLayout(btns)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        root.addWidget(line)

        credits = QLabel(
            f"<small>v{VERSION} · GPL-3.0-or-later<br>"
            "Inspired by work of Carlos Sonnenstein &amp; Oriol Lehmkuhl (PTeam).</small>"
        )
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credits.setWordWrap(True)
        credits.setStyleSheet("color: #777;")
        root.addWidget(credits)

        self._on_starless_toggled(self.chk_starless.isChecked())

    def _add_slider(self, parent_layout, label, lo, hi, value, tip):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        slider.setToolTip(tip)
        lbl = QLabel(f"{value / 100.0:.2f}")
        lbl.setFixedWidth(36)
        slider.valueChanged.connect(
            lambda v, L=lbl: (L.setText(f"{v / 100.0:.2f}"), self._mark_custom())
        )
        row.addWidget(slider)
        row.addWidget(lbl)
        parent_layout.addLayout(row)
        return slider, lbl

    def _show_help(self):
        box = QMessageBox(self)
        box.setWindowTitle(f"{TITLE} — Help")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(HELP_TEXT)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setMinimumWidth(480)
        box.exec()

    def _on_starless_toggled(self, checked: bool):
        self.guard_row_widget.setVisible(not checked)
        self._mark_custom()

    # --- Presets / params --------------------------------------------------

    def _mark_custom(self, *_):
        if getattr(self, "_applying_preset", False):
            return

    def _on_preset(self, index: int):
        self._apply_preset(index)

    def _apply_preset(self, index: int):
        self._applying_preset = True
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(index)
        self.combo_preset.blockSignals(False)

        # Starless is the normal workflow — keep it on across presets
        self.chk_starless.setChecked(True)

        if index == 0:  # Subtle
            self.slider_strength.setValue(28)
            self.combo_scale.setCurrentIndex(0)
            self.combo_guard.setCurrentIndex(1)
            self.slider_soft.setValue(60)
            self.slider_sky.setValue(45)
        elif index == 2:  # Dramatic
            self.slider_strength.setValue(70)
            self.combo_scale.setCurrentIndex(2)
            self.combo_guard.setCurrentIndex(2)
            self.slider_soft.setValue(35)
            self.slider_sky.setValue(20)
        else:  # Balanced
            self.slider_strength.setValue(45)
            self.combo_scale.setCurrentIndex(1)
            self.combo_guard.setCurrentIndex(1)
            self.slider_soft.setValue(50)
            self.slider_sky.setValue(35)

        self.chk_save_mask.setChecked(False)
        self.lbl_status.setText("Ready")
        self.progress.setValue(0)
        self._on_starless_toggled(True)
        self._applying_preset = False

    def _current_params(self) -> UmbraParams:
        return UmbraParams(
            strength=self.slider_strength.value() / 100.0,
            scale=self.combo_scale.currentIndex(),
            star_guard=self.combo_guard.currentIndex(),
            mask_softness=self.slider_soft.value() / 100.0,
            sky_protect=self.slider_sky.value() / 100.0,
            starless=self.chk_starless.isChecked(),
        )

    # --- Apply -------------------------------------------------------------

    def _set_busy(self, busy: bool):
        for w in (
            self.btn_apply, self.btn_reset, self.btn_help, self.combo_preset,
            self.chk_starless, self.slider_strength, self.combo_scale,
            self.combo_guard, self.slider_soft, self.slider_sky,
            self.chk_save_mask,
        ):
            w.setEnabled(not busy)
        if not busy:
            self._on_starless_toggled(self.chk_starless.isChecked())

    def _apply(self):
        img = self.ctx.get_image()
        if img is None:
            QMessageBox.critical(self, TITLE, "No image loaded in SASpro.")
            return

        data = np.array(img, dtype=np.float32, copy=True)
        params = self._current_params()

        self._set_busy(True)
        self.lbl_status.setText("Working...")
        self.progress.setValue(0)

        self._worker = UmbraWorker(data, params)
        self._worker.progress_update.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished_err.connect(self._on_err)
        self._worker.finished.connect(lambda: self._set_busy(False))
        self._worker.start()

    def _on_progress(self, message: str, pct: int):
        self.lbl_status.setText(message)
        self.progress.setValue(pct)

    def _on_ok(self, out, mask):
        self._last_mask = mask
        p = self._current_params()
        mode = "starless" if p.starless else f"stars={GUARD_LABELS[p.star_guard]}"
        try:
            self.ctx.set_image(
                out,
                step_name=f"DSE: str={p.strength:.2f} scale={SCALE_LABELS[p.scale]} {mode}",
            )

            if self.chk_save_mask.isChecked() and mask is not None:
                self._show_mask(mask)

            self.ctx.log(
                f"{TITLE}: applied (strength={p.strength:.2f}, "
                f"size={SCALE_LABELS[p.scale]}, {mode})"
            )
            self.lbl_status.setText("Applied")
            self.progress.setValue(100)
        except Exception as e:
            self.lbl_status.setText("Failed to write image")
            QMessageBox.critical(self, TITLE, f"Failed to apply to SASpro:\n{e}")

    def _on_err(self, error: str):
        self.lbl_status.setText("Failed")
        QMessageBox.critical(self, TITLE, f"Processing failed:\n{error}")

    def _show_mask(self, mask: np.ndarray):
        try:
            mask = mask.astype(np.float32, copy=False)
            if self.ctx.get_document(MASK_DOC_NAME):
                self.ctx.set_image_for(MASK_DOC_NAME, mask, step_name="Update DSE mask")
            else:
                self.ctx.open_new_document(mask, metadata=None, name=MASK_DOC_NAME)
            self.ctx.log(f"{TITLE}: mask shown in view '{MASK_DOC_NAME}'")
        except Exception as e:
            self.ctx.log(f"{TITLE}: could not show mask: {e}")


def run(ctx):
    """
    SASpro entry point.
    """
    w = UmbraDialog(ctx, parent=ctx.app)
    w.setModal(False)
    w.setWindowModality(Qt.WindowModality.NonModal)
    w.show()

    # Keep a reference on the context so Python doesn't GC the window
    try:
        setattr(ctx, "_dark_structures_window", w)
    except Exception:
        pass

    return w
