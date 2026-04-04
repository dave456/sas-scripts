#
# ***********************************************
#
# Original Author:  Nicolas CASTEL <nic.castel (at) gmail.com>
# Copyright (C) 2023 - Nicolas CASTEL
#
# Version 2.x Author: Carlo Mollicone - AstroBOH
# Copyright (C) 2025 - Carlo Mollicone
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Contact:
#   https://www.astroboh.it
#   https://www.facebook.com/carlo.mollicone.9
#
# SASPro Port by Gus
#
# ***********************************************
#
# Redistribution and modification are permitted under the terms
# of the GNU General Public License v3.0 or later, provided that:
#   1. All copyright notices and author attributions are preserved.
#   2. Any modifications to the code are clearly indicated.
#   3. Redistribution must comply with GPL-3.0 §4.
#   4. Derivative works are distributed under the same GPL-3.0-or-later license.
#
# This program is distributed WITHOUT ANY WARRANTY.
# See <https://www.gnu.org/licenses/gpl-3.0.html> for full terms.
#
# ***********************************************
# 
#
# Description:
# ------------------------------------------------------------------------------
# Project: Python SASPro script to run SCUNet denoiser via spandrel
#          using model from https://github.com/cszn/SCUNet
#          and https://ubersmooth.com/
#
#          GHS Stretch Engine based on the mathematical formulation by
#          Nightingale & Rowbottom (2021) — https://ghsastro.co.uk
#
#          Now supports:
#           - GUI Framework: PyQt6
#           - Single Image Processing
#           - Model Management: models are stored in a dedicated folder
#           - Mono image support
# ------------------------------------------------------------------------------
#
# Version History
# 1.0.3 - Original release by Nicolas CASTEL
# 2.0.0 - Ported to PyQt6 by Carlo Mollicone - AstroBOH
# 2.0.1 - SASPro Port
#       - Removed Siril-specific functions (sequence processing, overlay polygons)
#       - Adapted to SASPro scripting context API
#       - Uses SASPro bundled packages (torch, spandrel, PyQt6)
# 2.0.2 - Added explicit 'torchvision' dependency check (upstream v2.0.2)
# 2.0.3 - Added PreStretch for linear (unstretched) image support
#         New checkbox "Apply PreStretch" in the Parameters section of the GUI
#         Added DirectML (Windows) fallback device support
# 2.0.4 - Fix: Cannot set version_counter for inference tensor
#

VERSION = "2.0.4"

import sys
import os
import numpy as np
import urllib.request
import ssl
import math
import zipfile
import traceback

# ---------------------
#  SASPRO SCRIPT METADATA
# ---------------------
SCRIPT_NAME = "SCUNet Denoise"
SCRIPT_GROUP = "Denoise"

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QCheckBox, QMessageBox, QGroupBox, 
    QProgressBar, QDoubleSpinBox, QLineEdit, QFormLayout,
    QRadioButton, QSlider, QFrame, QStyle, QSizePolicy, QMainWindow
)
from PyQt6.QtGui import QCloseEvent, QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QSettings

# --- Torch & Spandrel ---
import subprocess

def ensure_package_installed(package_name, import_name=None):
    """
    Ensure a package is installed, installing it via pip if necessary.
    
    Args:
        package_name: The pip package name (e.g., 'spandrel')
        import_name: The import name if different from package name (optional)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_name, "--quiet"
        ])
        print(f"{package_name} installed successfully.")

# Ensure spandrel is installed
try:
    import torch
except ImportError:
    print("Error: PyTorch is not installed. Please install PyTorch first.")
    print("Visit https://pytorch.org/get-started/locally/ for installation instructions.")
    sys.exit(1)

try:
    from spandrel import ImageModelDescriptor, ModelLoader
except ImportError:
    print("Spandrel not found. Installing...")
    ensure_package_installed("spandrel")
    from spandrel import ImageModelDescriptor, ModelLoader
    print("Spandrel imported successfully.")

# --- Models List ---
# Format: [Name, URL, Description]
models_list = [
    ["SCUNet Color Real PSNR", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth", "Best all around model but can be too aggressive on stars"],
    ["SCUNet Color Real GAN", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth", "Less aggressive denoise"],
    ["SCUNet Color 15", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_15.pth", "Gaussian noise level 15"],
    ["SCUNet Color 25", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_25.pth", "Gaussian noise level 25"],
    ["SCUNet Color 50", "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_50.pth", "Gaussian noise level 50"],
    ["UberSmooth dso stars 0.1", "https://ubersmooth.com/uberSmooth-dso-stars-v0.1.zip", "Pretty good on stars but too aggressive on Hii regions"],
    ["UberSmooth dso stars 0.2", "https://ubersmooth.com/uberSmooth-dso-stars-v0.2.zip", "Not as aggressive as UberSmooth 0.1, but also not great"],
    ["UberSmooth planetary 0.1", "https://ubersmooth.com/uberSmooth-planetary-v0.1.zip", "Only denoise/deblur no extra star treatment"]
]

# ---------------------
#  THEME & STYLING
# ---------------------
DARK_STYLESHEET = """
QWidget { background-color: #2b2b2b; color: #e0e0e0; font-size: 10pt; }
QToolTip { background-color: #333333; color: #ffffff; border: 1px solid #88aaff; }
QGroupBox { border: 1px solid #444444; margin-top: 5px; font-weight: bold; border-radius: 4px; padding-top: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #88aaff; }
QLabel { color: #cccccc; }

QRadioButton, QCheckBox { color: #cccccc; spacing: 5px; }
QRadioButton::indicator, QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #666666; background: #3c3c3c; border-radius: 7px; }
QCheckBox::indicator { border-radius: 3px; }
QRadioButton::indicator:checked { background-color: #285299; border: 1px solid #88aaff; }
QCheckBox::indicator:checked { background-color: #285299; border: 1px solid #88aaff; }

QDoubleSpinBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox { background-color: #3c3c3c; color: #ffffff; border: 1px solid #555555; padding: 3px; border-radius: 3px; }
QComboBox:hover { border-color: #777777; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background-color: #3c3c3c; color: #ffffff; selection-background-color: #285299; border: 1px solid #555555; }

QSlider { min-height: 22px; }
QSlider::groove:horizontal { background: #444444; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal { background-color: #aaaaaa; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; border: 1px solid #555555; }
QSlider::handle:horizontal:hover { background-color: #ffffff; border: 1px solid #888888; }

QPushButton { background-color: #444444; color: #dddddd; border: 1px solid #666666; border-radius: 4px; padding: 6px; font-weight: bold;}
QPushButton:hover { background-color: #555555; border-color: #777777; }
QPushButton#ProcessButton { background-color: #285299; border: 1px solid #1e3f7a; }
QPushButton#ProcessButton:hover { background-color: #355ea1; }
QPushButton#CloseButton { background-color: #5a2a2a; border: 1px solid #804040; }
QPushButton#CloseButton:hover { background-color: #7a3a3a; }

QProgressBar { border: 1px solid #555555; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #285299; width: 10px; }
"""

# --- Core Logic Functions (Device & Tiling) ---

_dml_device = None

def _is_dml_device(device) -> bool:
    return _dml_device is not None and device == _dml_device

def get_device() -> torch.device:
    """
    Get the best available device for inference.
    Supported: NVIDIA CUDA, Apple MPS, Intel XPU, DirectML (Windows), CPU
    """
    global _dml_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")  # Intel Arc / XPU Support
    if sys.platform == "win32":
        try:
            import torch_directml
            _dml_device = torch_directml.device()
            return _dml_device
        except Exception:
            pass
    return torch.device("cpu")

def image_to_tensor(device: torch.device, img: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(img)
    return tensor.to(device)

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return (np.rollaxis(tensor.cpu().detach().numpy(), 1, 4).squeeze(0)).astype(np.float32)

def image_inference_tensor(model: ImageModelDescriptor, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.model(tensor)

def determine_optimal_tile_size(model, device, start_size=512):
    """
    Tries to run a dummy inference to find the maximum safe tile size.
    Returns the determined size (e.g., 512, 384, 256, or 128).
    """
    if device.type == 'cpu' or _is_dml_device(device):
        return 256  # Safe default for CPU/DirectML

    test_sizes = [512, 384, 256, 128]
    test_sizes = [s for s in test_sizes if s <= start_size]

    for size in test_sizes:
        try:
            dummy_input = torch.zeros(1, 3, size, size).to(device)
            with torch.no_grad():
                model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'xpu':
                torch.xpu.empty_cache()
            
            return size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'xpu':
                    torch.xpu.empty_cache()
                continue
            else:
                raise e
    
    return 128

def get_tile_weight(h, w, device):
    """
    Create a 2D weight mask that fades at the edges (from 0 to 1 and then back to 0).
    Use a linear ramp (Pyramid).
    """
    x = torch.linspace(0, 1, w, device=device)
    y = torch.linspace(0, 1, h, device=device)
    
    wx = torch.min(x, 1 - x) * 2
    wy = torch.min(y, 1 - y) * 2
    
    wx = torch.clamp(wx, min=0.1)
    wy = torch.clamp(wy, min=0.1)
    
    weight = wy.unsqueeze(1) * wx.unsqueeze(0)
    return weight.unsqueeze(0)

# --- PreStretch: GHS Stretch Engine ---
# Based on the mathematical formulation by Nightingale & Rowbottom (2021) — https://ghsastro.co.uk

_GHS_BG_TARGET = 0.25
_GHS_SHADOW_K = 0.0
_GHS_D_SCALE = 1.5
_GHS_SIGMA_ITER = 5

def _sigma_clip_stats(flat: np.ndarray, n_iter: int = _GHS_SIGMA_ITER) -> tuple:
    data = flat[flat > 1e-10].copy()
    if len(data) < 100:
        data = flat.copy()
    for _ in range(n_iter):
        med = np.median(data)
        mad = float(np.median(np.abs(data - med)))
        sigma = mad * 1.4826
        if sigma < 1e-12:
            break
        mask = np.abs(data - med) < 3.0 * sigma
        if mask.sum() < 50:
            break
        data = data[mask]
    med = float(np.median(data))
    mad = float(np.median(np.abs(data - med)))
    sigma = max(mad * 1.4826, 1e-12)
    return (med, sigma)

def _compute_ghs_params(hwc: np.ndarray, bg_target: float = _GHS_BG_TARGET, shadow_k: float = _GHS_SHADOW_K, d_scale: float = _GHS_D_SCALE) -> tuple:
    arr = np.asarray(hwc, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    hp_norm = float(np.nanmax(arr))
    if hp_norm < 1e-10:
        hp_norm = 1.0
    arr_norm = arr / hp_norm
    if arr_norm.shape[2] == 3:
        luminance = (arr_norm[:, :, 0] * 0.2126 + arr_norm[:, :, 1] * 0.7152 + arr_norm[:, :, 2] * 0.0722).ravel()
    else:
        luminance = arr_norm[:, :, 0].ravel()
    sky_med, sky_sigma = _sigma_clip_stats(luminance)
    LP = float(np.clip(shadow_k * sky_med, 0.0, sky_med * 0.9999))
    xp = sky_med - LP
    SP_base = max(xp, 1e-6)
    _bright_sky = xp > bg_target

    def _ghs_at_base(D_val: float) -> float:
        sp_search = 0.999 if _bright_sky else SP_base
        A_t = np.arcsinh(D_val * sp_search)
        B_t = np.arcsinh(D_val * (1.0 - sp_search)) + A_t
        if B_t < 1e-10:
            return sp_search
        return float((np.arcsinh(D_val * (xp - sp_search)) + A_t) / B_t)

    D_lo, D_hi = (1.0, 1e8)
    for _ in range(40):
        D_mid = np.sqrt(D_lo * D_hi)
        if _bright_sky:
            if _ghs_at_base(D_mid) > bg_target:
                D_lo = D_mid
            else:
                D_hi = D_mid
        elif _ghs_at_base(D_mid) < bg_target:
            D_lo = D_mid
        else:
            D_hi = D_mid
    D_base = float(np.sqrt(D_lo * D_hi))
    D = D_base * d_scale
    _y_max_approx = float(np.arcsinh(D * xp) / max(np.arcsinh(D), 1e-10))
    _sp_feasible = _y_max_approx >= bg_target

    def y_at_sp(test_sp: float) -> float:
        A_t = np.arcsinh(D * test_sp)
        B_t = np.arcsinh(D * (1.0 - test_sp)) + A_t
        if B_t < 1e-10:
            return 0.0
        return (np.arcsinh(D * (xp - test_sp)) + A_t) / B_t

    if _sp_feasible:
        SP_lo, SP_hi = (1e-6, 0.999)
        for _ in range(40):
            SP_mid = (SP_lo + SP_hi) / 2.0
            if y_at_sp(SP_mid) > bg_target:
                SP_lo = SP_mid
            else:
                SP_hi = SP_mid
        SP = float((SP_lo + SP_hi) / 2.0)
    else:
        SP = SP_base
    A = float(np.arcsinh(D * SP))
    B = float(np.arcsinh(D * (1.0 - SP)) + A)
    master_param = {'LP': LP, 'SP': SP, 'D': D, 'A': A, 'B': B, 'hp_norm': hp_norm}
    params = [master_param] * arr.shape[2]
    return (params, hp_norm)

def _apply_stretch(hwc: np.ndarray, params: list) -> np.ndarray:
    arr = np.asarray(hwc, dtype=np.float64)
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[:, :, np.newaxis]
    hp_norm = params[0].get('hp_norm', 1.0)
    arr_norm = arr / hp_norm
    out = np.empty_like(arr_norm)
    for c, p in enumerate(params):
        LP, SP, D, A, B = (p['LP'], p['SP'], p['D'], p['A'], p['B'])
        xp = arr_norm[:, :, c] - LP
        out[:, :, c] = (np.arcsinh(D * (xp - SP)) + A) / B
    return out[:, :, 0] if is_2d else out

def _apply_destretch(hwc: np.ndarray, params: list) -> np.ndarray:
    arr = np.asarray(hwc, dtype=np.float64)
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[:, :, np.newaxis]
    hp_norm = params[0].get('hp_norm', 1.0)
    out = np.empty_like(arr)
    for c, p in enumerate(params):
        LP, SP, D, A, B = (p['LP'], p['SP'], p['D'], p['A'], p['B'])
        y = arr[:, :, c]
        xp = np.sinh(np.clip(y * B - A, -500.0, 500.0)) / D + SP
        x_norm = xp + LP
        out[:, :, c] = np.clip(x_norm * hp_norm, 0.0, hp_norm)
    return out[:, :, 0] if is_2d else out


def tile_process(device: torch.device, model: ImageModelDescriptor, data: np.ndarray, scale, tile_size, yield_extra_details=False, apply_prestretch=False, precomputed_ghs_params=None):
    """
    Process data [height, width, channel] into tiles of size [tile_size, tile_size, channel],
    feed them one by one into the model, then yield the resulting output tiles.
    Uses stride-based overlapping tiling to avoid seam artifacts.
    """
    tile_pad = 144

    # [height, width, channel] -> [1, channel, height, width]
    data = np.rollaxis(data, 2, 0)
    data = np.expand_dims(data, axis=0)

    batch, channel, height, width = data.shape

    # Stride-based overlapping tiling: 75% overlap eliminates seam artifacts
    stride = tile_size * 3 // 4

    def make_starts(dim, ts, st):
        starts = list(range(0, max(1, dim - ts), st))
        if not starts or starts[-1] + ts < dim:
            starts.append(max(0, dim - ts))
        return starts

    # Compute global GHS params once if prestretch is enabled
    global_ghs_params = None
    if apply_prestretch:
        if precomputed_ghs_params is not None:
            global_ghs_params = precomputed_ghs_params
        else:
            full_img_hwc = data[0].transpose(1, 2, 0).astype(np.float64)
            global_ghs_params, _ = _compute_ghs_params(full_img_hwc)

    h_starts = make_starts(height, tile_size, stride)
    w_starts = make_starts(width, tile_size, stride)
    total_tiles = len(h_starts) * len(w_starts)
    tile_count = 0

    for input_start_y in h_starts:
        for input_start_x in w_starts:
            tile_count += 1
            input_end_x = input_start_x + tile_size
            input_end_y = input_start_y + tile_size

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = tile_size
            input_tile_height = tile_size

            input_tile = data[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad].astype(np.float32)

            if apply_prestretch:
                tile_hwc = input_tile[0].transpose(1, 2, 0).astype(np.float64)
                tile_stretched = _apply_stretch(tile_hwc, global_ghs_params)
                input_tile = tile_stretched.astype(np.float32).transpose(2, 0, 1)[np.newaxis]

            output_tile = image_inference_tensor(model, image_to_tensor(device, input_tile))
            progress = tile_count / total_tiles

            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            output_tile = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
            output_tile = tensor_to_image(output_tile)

            if apply_prestretch:
                out_hwc = output_tile.astype(np.float64)
                out_hwc = _apply_destretch(out_hwc, global_ghs_params)
                output_tile = out_hwc.astype(np.float32)

            if yield_extra_details:
                yield (output_tile, input_start_y, input_start_x, input_tile_width, input_tile_height, progress)
            else:
                yield output_tile
    yield None


def process_image_buffer(image_data, model, device, strength, tile_size, progress_callback=None, apply_prestretch=False, precomputed_ghs_params=None):
    """
    Process an image buffer through the SCUNet model.

    Args:
        image_data: Input image as numpy array (H,W,C) or (H,W) for mono, normalized 0-1 float32
        model: Loaded SCUNet model
        device: Torch device
        strength: Blend strength 0-1
        tile_size: Tile size for processing
        progress_callback: Optional callback for progress updates
        apply_prestretch: Apply GHS pre-stretch for linear (unstretched) images
        precomputed_ghs_params: Pre-computed GHS parameters (optional, computed automatically if None)

    Returns:
        Processed image as numpy array in same format as input
    """
    original_dtype = image_data.dtype

    # 1. Ensure float32 normalized 0-1
    if original_dtype == np.uint8:
        pixel_data = image_data.astype(np.float32) / 255.0
        divisor = 255.0
    elif original_dtype == np.uint16:
        actual_max = image_data.max()
        divisor = 255.0 if actual_max <= 255 else 65535.0
        pixel_data = image_data.astype(np.float32) / divisor
    else:
        pixel_data = image_data.astype(np.float32)
        divisor = 1.0

    # 2. Handle mono images -> RGB (model requires 3 channels)
    is_mono = False
    if pixel_data.ndim == 2:
        is_mono = True
        # (H, W) -> (3, H, W)
        pixel_data = np.stack((pixel_data,)*3, axis=0)
    elif pixel_data.ndim == 3 and pixel_data.shape[2] == 1:
        # (H, W, 1) -> (3, H, W)
        is_mono = True
        pixel_data = np.repeat(pixel_data.transpose(2, 0, 1), 3, axis=0)
    elif pixel_data.ndim == 3 and pixel_data.shape[2] == 3:
        # (H, W, C) -> (C, H, W)
        pixel_data = pixel_data.transpose(2, 0, 1)
    elif pixel_data.ndim == 3 and pixel_data.shape[0] == 1:
        is_mono = True
        pixel_data = np.repeat(pixel_data, 3, axis=0)

    c, h, w = pixel_data.shape

    # (C, H, W) -> (H, W, C) for tile_process
    pixel_data_hwc = np.transpose(pixel_data, (1, 2, 0))

    # Accumulation buffers
    output_sum = torch.zeros((c, h, w), dtype=torch.float32, device='cpu')
    output_weight = torch.zeros((c, h, w), dtype=torch.float32, device='cpu')

    base_weight_mask = get_tile_weight(tile_size, tile_size, 'cpu')

    scale = 1  # SCUNet does not upscale

    for i, tile_info in enumerate(tile_process(device, model, pixel_data_hwc, scale, tile_size, yield_extra_details=True, apply_prestretch=apply_prestretch, precomputed_ghs_params=precomputed_ghs_params)):
        if tile_info is None:
            break

        tile_data_numpy, y_start, x_start, _, _, p = tile_info

        if progress_callback:
            if progress_callback(p) is False:
                return None

        tile_tensor = torch.from_numpy(tile_data_numpy.transpose(2, 0, 1))
        c_real, h_real, w_real = tile_tensor.shape

        if h_real <= 0 or w_real <= 0:
            continue

        y_end = y_start + h_real
        x_end = x_start + w_real

        y_end_safe = min(y_end, h)
        x_end_safe = min(x_end, w)
        y_start_safe = max(0, y_start)
        x_start_safe = max(0, x_start)

        write_h = y_end_safe - y_start_safe
        write_w = x_end_safe - x_start_safe

        if write_h <= 0 or write_w <= 0:
            continue

        if write_h != h_real or write_w != w_real:
            tile_tensor = tile_tensor[:, :write_h, :write_w]
            h_real = write_h
            w_real = write_w

        if h_real == tile_size and w_real == tile_size:
            mask = base_weight_mask
        else:
            mask = get_tile_weight(h_real, w_real, 'cpu')

        output_sum[:, y_start_safe:y_end_safe, x_start_safe:x_end_safe] += tile_tensor * mask
        output_weight[:, y_start_safe:y_end_safe, x_start_safe:x_end_safe] += mask

    # Final normalization
    output_image_tensor = output_sum / (output_weight + 1e-8)
    output_image = output_image_tensor.numpy()

    # Mono restore
    if is_mono:
        output_image = output_image[0, :, :]
    else:
        # (C, H, W) -> (H, W, C)
        output_image = output_image.transpose(1, 2, 0)

    # De-normalization
    final_dtype = original_dtype
    if original_dtype == np.uint8:
        output_image = np.clip(output_image, 0, 1) * 255.0
        output_image = output_image.astype(np.uint8)
    elif original_dtype == np.uint16:
        output_image = np.clip(output_image, 0, 1) * divisor
        output_image = output_image.astype(np.uint16)
    else:
        output_image = np.clip(output_image, 0, 1).astype(np.float32)

    # Blend with original
    if strength != 1.0:
        blended = output_image * strength + image_data * (1 - strength)
        return blended.astype(final_dtype)
    else:
        return output_image


# --- Worker Thread ---
class ProcessingWorker(QObject):
    """
    A worker that performs model download and processing
    in a separate thread to avoid blocking the GUI.
    """
    finished = pyqtSignal(object)  # Returns processed image or None
    progress_update = pyqtSignal(int, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, ctx, image_data, params):
        super().__init__()
        self.ctx = ctx
        self.image_data = image_data
        self.params = params
        self._is_running = True

    def run(self):
        try:
            model_url = self.params['model_url']
            strength = self.params['strength']

            # 1. Setup Model Directory
            self.progress_update.emit(0, "Checking Model...")
            
            # Use user's home directory for model storage
            user_dir = os.path.expanduser("~")
            models_dir = os.path.join(user_dir, ".saspro", "scunet_models")
            
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            model_filename = os.path.basename(model_url)
            modelpath = os.path.join(models_dir, model_filename)

            def download_progress_hook(block_num, block_size, total_size):
                if not self._is_running:
                    raise Exception("Download cancelled")
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = int((downloaded / total_size) * 100)
                    if percent % 2 == 0: 
                        self.progress_update.emit(percent, f"Downloading Model: {percent}%")
                else:
                    self.progress_update.emit(0, "Downloading Model... (size unknown)")

            # Download if not exists
            if not os.path.isfile(modelpath):
                self.ctx.log(f"Downloading model to: {modelpath}")
                ssl._create_default_https_context = ssl._create_stdlib_context
                
                try:
                    urllib.request.urlretrieve(model_url, modelpath, reporthook=download_progress_hook)
                    self.ctx.log("Model download completed.")
                except Exception as e:
                    if os.path.exists(modelpath):
                        os.remove(modelpath)
                    raise e
            else:
                self.ctx.log(f"Using existing model at: {modelpath}")
            
            # ZIP Management (UberSmooth)
            if zipfile.is_zipfile(modelpath):
                with zipfile.ZipFile(modelpath, 'r') as zip_ref:
                    zip_ref.extractall(models_dir)
                    modelpath = modelpath.replace(".zip", ".pth")

            # 2. Load Model
            self.progress_update.emit(0, "Loading Model into Memory...")
            device = get_device()

            # Log device info
            self.ctx.log("------ Hardware Info ------")
            cuda_available = torch.cuda.is_available()
            self.ctx.log(f"CUDA Available: {cuda_available}")
            if cuda_available:
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    self.ctx.log(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            
            mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            self.ctx.log(f"MPS (Apple) Available: {mps_available}")
            
            xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
            self.ctx.log(f"XPU (Intel) Available: {xpu_available}")

            dml_available = _is_dml_device(device)
            self.ctx.log(f"DirectML (Windows) Available: {dml_available}")

            self.ctx.log(f"Active Device: {'DirectML' if dml_available else device.type.upper()}")
            self.ctx.log("---------------------------")

            if device.type == 'cuda':
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown NVIDIA GPU"
                self.ctx.log(f"Acceleration: CUDA ({device_name})")
            elif device.type == 'mps':
                self.ctx.log("Acceleration: Apple Metal Performance Shaders (MPS)")
            elif device.type == 'xpu':
                self.ctx.log("Acceleration: Intel XPU (Arc GPU detected)")
            elif dml_available:
                self.ctx.log("Acceleration: DirectML (Windows GPU fallback)")
            else:
                self.ctx.log("Acceleration: CPU (No GPU detected)")
            
            model = ModelLoader().load_from_file(str(modelpath)).eval().to(device)

            # Check architecture
            architecture_name = type(model.model).__name__
            
            if "SCUNet" not in architecture_name:
                raise RuntimeError(
                    f"Invalid model selected. Expected a SCUNet model, but detected: '{architecture_name}'.\n"
                    "Please select a valid SCUNet .pth file."
                )

            assert isinstance(model, ImageModelDescriptor)

            # Determine tile size
            req_tile = self.params['tile_size']
            
            if req_tile == "Auto":
                self.progress_update.emit(0, "Auto-tuning Tile Size...")
                final_tile_size = determine_optimal_tile_size(model, device)
                self.ctx.log(f"Auto-Tuning: Selected Tile Size {final_tile_size}px")
            else:
                final_tile_size = req_tile
                self.ctx.log(f"Manual Tile Size: {final_tile_size}px")

            # 3. Process image
            def callback(tile_p):
                if not self._is_running:
                    return False
                self.progress_update.emit(int(tile_p * 100), "Denoising Image...")
                return True

            apply_prestretch = self.params.get('apply_prestretch', False)
            processed_data = process_image_buffer(
                self.image_data, model, device, strength, final_tile_size, callback,
                apply_prestretch=apply_prestretch
            )
            
            if processed_data is None:
                self.ctx.log("Processing cancelled.")
                self.finished.emit(None)
                return

            self.progress_update.emit(100, "Done.")
            self.finished.emit(processed_data)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False


# --- Main GUI Class ---
class ScunetWindow(QMainWindow):
    def __init__(self, ctx, qt_app):
        super().__init__()
        self.ctx = ctx
        self.app = qt_app
        
        self.setWindowTitle(f"SCUNet Denoise - v{VERSION}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        self.settings = QSettings("SASPro", "SCUNetDenoise")
        
        self.linear_cache = None
        self.thread = None
        self.worker = None
        
        # Log header
        header_msg = (
            "##############################################\n"
            "# SCUNet Denoise (SASPro Port)\n"
            "# Original by Nicolas CASTEL & Carlo Mollicone\n"
            "##############################################"
        )
        self.ctx.log(header_msg)
        
        self.setup_ui()
        self.center_window()
        self.cache_input()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        
        self.app.setStyle("Fusion")
        self.setStyleSheet(DARK_STYLESHEET)
        
        # Header
        head_title = QLabel(f"SCUNet Denoise v{VERSION}")
        head_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        head_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #88aaff;")
        layout.addWidget(head_title)
        
        # Credits
        lbl_credits = QLabel(
            "<span style='color:#f4d742;'><b>Original by Nicolas CASTEL</b></span><br>"
            "Refactoring by Carlo Mollicone AstroBOH"
        )
        lbl_credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_credits.setStyleSheet("color: #888; font-size: 10px; margin-bottom: 5px;")
        layout.addWidget(lbl_credits)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # Instructions
        lbl_instructions = QLabel(
            "<span style='color:#d0d0d0;'><b>SCUNet denoiser</b></span> works best on fully processed "
            "and stretched non-linear images.<br>"
            "Run as the last step before publishing.<br>"
            "Ensure your image is in 16-bit or 32-bit format for optimal results.<br>"
            "Make sure to click on Reload Input to cache the current active image before processing."
        )
        lbl_instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_instructions.setStyleSheet("color: #888; font-size: 10px; margin-bottom: 5px;")
        lbl_instructions.setWordWrap(True)
        layout.addWidget(lbl_instructions)

        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line2)

        # 1. Model Selection
        gb_model = QGroupBox("Model Selection")
        layout_model = QVBoxLayout()
        
        self.model_buttons = []
        
        for i, m in enumerate(models_list):
            rb = QRadioButton(m[0])
            rb.setToolTip(m[2])
            rb.setProperty("url", m[1])
            rb.setProperty("description", m[2])
            
            if i == 0:
                rb.setChecked(True)
            
            # Connect to update description when selected
            rb.toggled.connect(self.update_model_description)
            
            layout_model.addWidget(rb)
            self.model_buttons.append(rb)
        
        # Model description label
        self.lbl_model_desc = QLabel(models_list[0][2])
        self.lbl_model_desc.setWordWrap(True)
        self.lbl_model_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_model_desc.setStyleSheet(
            "color: #aaccff; font-style: italic; font-size: 9pt; "
            "padding: 6px; background-color: #333333; border-radius: 4px; margin-top: 4px;"
        )
        layout_model.addWidget(self.lbl_model_desc)
        
        gb_model.setLayout(layout_model)
        layout.addWidget(gb_model)

        # 2. Parameters
        gb_params = QGroupBox("Parameters")
        layout_params = QVBoxLayout()
        
        # Strength Slider
        h_slider = QHBoxLayout()
        self.slider_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_strength.setRange(0, 100)
        self.slider_strength.setValue(50)

        self.lbl_strength_val = QLabel("0.50")
        self.slider_strength.valueChanged.connect(
            lambda val: self.lbl_strength_val.setText(f"{val/100:.2f}")
        )
        
        h_slider.addWidget(QLabel("Strength:"))
        h_slider.addWidget(self.slider_strength)
        h_slider.addWidget(self.lbl_strength_val)
        layout_params.addLayout(h_slider)

        # Tile Size
        h_tile = QHBoxLayout()
        self.combo_tile = QComboBox()
        self.combo_tile.addItems(["Auto", "512", "384", "256", "128"])
        self.combo_tile.setToolTip(
            "Tile size for processing.\n"
            "'Auto' tests VRAM to find the best size.\n"
            "Lower values save memory but may be slower."
        )
        
        h_tile.addWidget(QLabel("Tile Size:"))
        h_tile.addWidget(self.combo_tile)
        layout_params.addLayout(h_tile)

        gb_params.setLayout(layout_params)
        layout.addWidget(gb_params)

        # 3. Linear Image Options
        gb_linear = QGroupBox("Linear Image Options")
        layout_linear = QVBoxLayout()

        self.chk_prestretch = QCheckBox("Apply PreStretch (for linear/unstretched images)")
        self.chk_prestretch.setChecked(False)
        self.chk_prestretch.toggled.connect(self._on_prestretch_toggled)
        self.chk_prestretch.setToolTip(
            "Enable this if your image is still in linear (unstretched) state.\n"
            "Leave OFF for standard non-linear (already stretched) images."
        )
        layout_linear.addWidget(self.chk_prestretch)

        gb_linear.setLayout(layout_linear)
        layout.addWidget(gb_linear)

        # Progress
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(10)

        # Buttons
        btn_layout = QHBoxLayout()
        
        b_reload = QPushButton("Reload Input")
        b_reload.setToolTip("Reload image from active document.")
        b_reload.clicked.connect(self.cache_input)
        
        b_proc = QPushButton("PROCESS")
        b_proc.setObjectName("ProcessButton")
        b_proc.setToolTip("Apply denoise to the image.")
        b_proc.clicked.connect(self.start_processing)
        
        b_close = QPushButton("Close")
        b_close.setObjectName("CloseButton")
        b_close.clicked.connect(self.close)
        
        btn_layout.addWidget(b_reload)
        btn_layout.addStretch()
        btn_layout.addWidget(b_proc)
        btn_layout.addWidget(b_close)
        layout.addLayout(btn_layout)
        
        self.btn_proc = b_proc
        self.btn_close = b_close

    def center_window(self):
        self.setMinimumWidth(560)
        self.adjustSize()
        screen = self.app.primaryScreen()
        if screen:
            frame_geo = self.frameGeometry()
            frame_geo.moveCenter(screen.availableGeometry().center())
            self.move(frame_geo.topLeft())

    def update_model_description(self, checked):
        """Update the model description label when a radio button is selected."""
        if checked:
            sender = self.sender()
            if sender:
                description = sender.property("description")
                if description:
                    self.lbl_model_desc.setText(description)

    def _on_prestretch_toggled(self, checked):
        """Adjust default strength when switching between linear/non-linear mode."""
        self.slider_strength.setValue(70 if checked else 50)

    def cache_input(self):
        try:
            self.lbl_status.setText("Caching Input...")
            self.app.processEvents()
            
            img = self.ctx.get_image()
            if img is None:
                self.lbl_status.setText("Error: No active image.")
                return
            
            self.linear_cache = img.copy()
            self.lbl_status.setText("Input Cached. Ready to process.")
            self.ctx.log("SCUNet: Input cached successfully.")
            
        except Exception as e:
            self.lbl_status.setText(f"Error: {str(e)}")
            self.ctx.log(f"SCUNet Error: {str(e)}")

    def start_processing(self):
        if self.linear_cache is None:
            QMessageBox.warning(self, "Warning", "No image cached. Please reload input.")
            return
        
        # Get selected model URL
        selected_url = ""
        for rb in self.model_buttons:
            if rb.isChecked():
                selected_url = rb.property("url")
                break
        
        if not selected_url:
            QMessageBox.warning(self, "Warning", "Please select a model.")
            return

        # Get parameters
        strength_val = self.slider_strength.value() / 100.0
        tile_choice = self.combo_tile.currentText()
        tile_param = "Auto" if tile_choice == "Auto" else int(tile_choice)

        params = {
            'model_url': selected_url,
            'strength': strength_val,
            'tile_size': tile_param,
            'apply_prestretch': self.chk_prestretch.isChecked(),
        }

        # Setup thread
        self.thread = QThread()
        self.worker = ProcessingWorker(self.ctx, self.linear_cache, params)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.finished.connect(self.process_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # UI state
        self.lbl_status.setText("Starting...")
        self.btn_proc.setEnabled(False)
        self.btn_close.setText("Cancel")

        self.thread.start()

    def update_progress(self, val, text):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(text)

    def handle_error(self, msg):
        QMessageBox.critical(self, "Processing Error", msg)
        self.lbl_status.setText("Error occurred.")
        self.btn_proc.setEnabled(True)
        self.btn_close.setText("Close")

    def process_finished(self, result_img):
        self.btn_proc.setEnabled(True)
        self.btn_close.setText("Close")
        
        if result_img is not None:
            # Set image via SASPro context (handles undo automatically)
            self.ctx.set_image(result_img, step_name=f"SCUNet Denoise v{VERSION}")
            self.lbl_status.setText("Complete. Image updated.")
            self.ctx.log(f"SCUNet v{VERSION}: Denoise applied successfully.")
        else:
            self.lbl_status.setText("Processing cancelled.")

    def closeEvent(self, event: QCloseEvent):
        # Stop worker if running
        if self.worker:
            self.worker.stop()
        
        if self.thread:
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    if not self.thread.wait(5000):  # Wait up to 5 seconds
                        self.ctx.log("SCUNet: Warning - thread did not stop gracefully.")
            except RuntimeError:
                pass
        
        # Clean up global reference
        global _scunet_window
        _scunet_window = None
        
        self.ctx.log("SCUNet Denoise: Window closed.")
        event.accept()


# =============================================================================
#  SASPRO SCRIPT ENTRYPOINT
# =============================================================================

# Global reference to keep the window alive while SASPro runs
_scunet_window = None

def run(ctx):
    """
    SASPro script entrypoint.
    Launches the SCUNet Denoise GUI.
    """
    global _scunet_window
    
    try:
        # Check for active image
        img = ctx.get_image()
        if img is None:
            ctx.log("SCUNet Error: No active image. Please open an image first.")
            return
        
        # Get the existing QApplication instance (SASPro provides this)
        app = QApplication.instance()
        if not app:
            # Fallback: should not happen when running inside SASPro
            ctx.log("SCUNet Warning: No QApplication found, creating one.")
            app = QApplication(sys.argv)
        
        # Close existing window if one is already open
        if _scunet_window is not None:
            try:
                _scunet_window.close()
            except RuntimeError:
                pass  # Window was already deleted
            _scunet_window = None
        
        # Create and show the GUI
        # Store in global to prevent garbage collection
        _scunet_window = ScunetWindow(ctx, app)
        _scunet_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        _scunet_window.show()
        
        # Do NOT call app.exec() - SASPro already runs the Qt event loop
        # The window will be handled by SASPro's existing event loop
        
    except Exception as e:
        ctx.log(f"SCUNet Error: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("SCUNet Denoise v" + VERSION)
    print("This script must be run from within SASPro.")
    print("Please open SASPro and run this script from the Scripts menu.")
