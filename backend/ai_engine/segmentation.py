"""
Body Segmentation Module
========================
Removes background from body images using rembg (U2-Net) and optionally
MediaPipe's segmentation mask for body-only isolation.
"""

import cv2
import numpy as np
from PIL import Image
import io
import os


def remove_background(image_path: str, output_path: str = None) -> np.ndarray:
    """
    Remove background from the body image using rembg (U2-Net model).

    Args:
        image_path:  Path to input image.
        output_path: If provided, saves the BGRA result image there.

    Returns:
        BGRA NumPy array (alpha channel = 0 for background, 255 for body).
    """
    try:
        from rembg import remove
        with open(image_path, "rb") as f:
            raw = f.read()
        result_bytes = remove(raw)
        result_pil   = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        result_bgra  = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGBA2BGRA)
    except ImportError:
        # Fallback: use GrabCut if rembg not installed
        result_bgra = _grabcut_segmentation(image_path)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result_bgra)

    return result_bgra


def _grabcut_segmentation(image_path: str) -> np.ndarray:
    """
    Fallback background removal using OpenCV GrabCut.
    Less accurate than rembg but has no extra dependencies.
    """
    bgr  = cv2.imread(image_path)
    h, w = bgr.shape[:2]

    # Initialise mask and models for GrabCut
    mask   = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Assume subject is roughly centred with 10% border margin
    rect = (w // 10, h // 10, w - w // 5, h - h // 5)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Build BGRA image
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = fg_mask
    return bgra


def get_body_mask(image_path: str) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) of the body region.

    Tries rembg first (U2-Net, best quality), then GrabCut fallback.

    Args:
        image_path: Path to body image.

    Returns:
        Binary mask as (H, W) uint8 numpy array.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = bgr.shape[:2]

    # ── Try rembg (best quality) ───────────────────────────────────────────────
    try:
        from rembg import remove
        with open(image_path, "rb") as f:
            raw = f.read()
        result_bytes = remove(raw)
        result_pil   = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
        result_rgba  = np.array(result_pil)
        mask = result_rgba[:, :, 3]  # alpha channel = body mask
        # Resize to match original if needed
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8) * 255
    except Exception:
        pass

    # ── Fallback: GrabCut ─────────────────────────────────────────────────────
    return _grabcut_segmentation_mask(bgr)


def _grabcut_segmentation_mask(bgr: np.ndarray) -> np.ndarray:
    """GrabCut-based body segmentation, returns (H,W) uint8 mask."""
    h, w  = bgr.shape[:2]
    mask  = np.zeros((h, w), np.uint8)
    bgd   = np.zeros((1, 65), np.float64)
    fgd   = np.zeros((1, 65), np.float64)

    # Rect: assume person is centred, occupying middle 80% of image
    margin_x = w // 10
    margin_y = h // 15
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    # Smooth edges
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    return (fg_mask > 127).astype(np.uint8) * 255


def apply_mask_to_image(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to a BGR image, returning BGRA with transparent background.
    """
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra
