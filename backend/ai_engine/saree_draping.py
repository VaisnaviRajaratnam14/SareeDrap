"""
Saree Draping Module
====================
Handles:
 - Saree feature extraction (dominant colour, border detection, texture)
 - Pleat generation
 - Pallu generation
 - Overlay of saree on body using geometric transformations
 - Style-specific draping: nivi, bridal, hanging, gujarati
"""

import cv2
import numpy as np
from typing import Optional
from sklearn.cluster import KMeans   # type: ignore


# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_saree_features(image_path: str, fabric_type: str = "silk") -> dict:
    """
    Analyse a saree image to extract:
     - dominant_color (hex)
     - border_color   (hex, bottom strip)
     - border_width   (pixels)
     - texture_map    (base64 encoded 128x128 patch)
     - fabric_type
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Saree image not found: {image_path}")

    h, w = bgr.shape[:2]

    dominant_color = _get_dominant_color(bgr)
    border_color, border_width = _detect_border(bgr)
    texture_b64    = _extract_texture_patch(bgr)

    return {
        "dominant_color":  dominant_color,
        "border_color":    border_color,
        "border_width":    border_width,
        "texture_patch":   texture_b64,
        "fabric_type":     fabric_type,
        "image_size":      {"width": w, "height": h},
    }


def _get_dominant_color(bgr: np.ndarray, k: int = 3) -> str:
    """Return the most dominant colour of the saree as a hex string."""
    small  = cv2.resize(bgr, (80, 80)).reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(small)
    # Most frequent cluster
    counts  = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
    b, g, r  = int(dominant[0]), int(dominant[1]), int(dominant[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def _detect_border(bgr: np.ndarray, strip_frac: float = 0.08) -> tuple:
    """
    Detect the saree border by examining the bottom strip.
    Returns (border_color_hex, border_width_px).
    """
    h, w = bgr.shape[:2]
    strip_h = max(int(h * strip_frac), 10)
    border_strip = bgr[h - strip_h: h, :, :]
    avg = border_strip.reshape(-1, 3).mean(axis=0).astype(int)
    b, g, r = int(avg[0]), int(avg[1]), int(avg[2])
    hex_color = f"#{r:02X}{g:02X}{b:02X}"
    return hex_color, strip_h


def _extract_texture_patch(bgr: np.ndarray, size: int = 128) -> str:
    """Extract a 128×128 texture patch from the centre and return as base64."""
    import base64
    h, w = bgr.shape[:2]
    cy, cx = h // 2, w // 2
    half   = size // 2
    patch  = bgr[
        max(0, cy - half): cy + half,
        max(0, cx - half): cx + half,
    ]
    patch = cv2.resize(patch, (size, size))
    _, buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


# ── Draping Engine ─────────────────────────────────────────────────────────────

def drape_saree(
    body_bgra: np.ndarray,
    saree_bgr: np.ndarray,
    anchors: dict,
    draping_style: str = "nivi",
    fabric_type: str = "silk",
) -> np.ndarray:
    _styles  = {"nivi": _drape_nivi, "bridal": _drape_bridal,
                "hanging": _drape_hanging, "gujarati": _drape_gujarati}
    style_fn = _styles.get(draping_style, _drape_nivi)
    result   = style_fn(body_bgra.copy(), saree_bgr, anchors)
    result   = _simulate_fabric(result, fabric_type)
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _a(anchors, key, default):
    v = anchors.get(key, default)
    if isinstance(v, dict):
        return float(v["x"]), float(v["y"])
    return float(v)


def _drop_shadow(canvas: np.ndarray, mask_w: np.ndarray,
                 offset_x: int = 6, offset_y: int = 8,
                 blur_r: int = 15, strength: float = 0.45) -> np.ndarray:
    """Paint a soft drop shadow on canvas under the area defined by mask_w."""
    shadow = cv2.GaussianBlur(mask_w, (blur_r | 1, blur_r | 1), 0)
    # Shift shadow down-right
    M_shift = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    ch, cw  = canvas.shape[:2]
    shadow  = cv2.warpAffine(shadow, M_shift, (cw, ch))
    shadow  = (shadow * strength)[:, :, np.newaxis]
    result  = canvas.copy()
    bg      = result[:, :, :3].astype(np.float32)
    darkened = np.clip(bg * (1.0 - shadow), 0, 255).astype(np.uint8)
    result[:, :, :3] = darkened
    return result


def _warp_quad(src_bgr: np.ndarray, dst_pts: np.ndarray,
               canvas: np.ndarray, alpha: float = 0.85,
               shadow: bool = False) -> np.ndarray:
    """Perspective-warp src_bgr into the quadrilateral dst_pts on canvas."""
    sh, sw = src_bgr.shape[:2]
    ch, cw = canvas.shape[:2]
    src_pts = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst     = np.float32(dst_pts)
    M       = cv2.getPerspectiveTransform(src_pts, dst)
    warped  = cv2.warpPerspective(src_bgr, M, (cw, ch),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_TRANSPARENT)
    # Build mask from warped area
    mask_src = np.ones((sh, sw), dtype=np.uint8) * 255
    mask_w   = cv2.warpPerspective(mask_src, M, (cw, ch)).astype(np.float32) / 255.0

    # Feather mask edges (wider feather = softer edge)
    ks = 11
    mask_smooth = cv2.GaussianBlur(mask_w, (ks, ks), 0)
    mask_blend  = (mask_smooth * alpha)[:, :, np.newaxis]

    # Optional drop shadow drawn first (under the fabric)
    if shadow:
        canvas = _drop_shadow(canvas, mask_w)

    result = canvas.copy()
    bg_bgr = result[:, :, :3].astype(np.float32)
    ov_bgr = warped.astype(np.float32)
    blended = (ov_bgr * mask_blend + bg_bgr * (1.0 - mask_blend)).astype(np.uint8)
    result[:, :, :3] = blended
    return result


def _fan_pleats(canvas: np.ndarray, saree_bgr: np.ndarray,
                wx: float, wy: float,
                sh_width: float, body_h: float,
                num_pleats: int = 7,
                pleat_alpha: float = 0.88) -> np.ndarray:
    """
    Render fan-shaped pleats at the waist center.
    Each pleat is a narrow trapezoid with alternating light/dark for fold depth.
    """
    pleat_w_top = int(sh_width / num_pleats)
    pleat_w_bot = int(pleat_w_top * 0.78)      # tighter taper for realistic folds
    pleat_h     = int(body_h * 0.65)            # taller pleats (was 0.55)
    total_w_top = pleat_w_top * num_pleats
    x_start     = int(wx - total_w_top / 2)

    sh, sw = saree_bgr.shape[:2]

    for i in range(num_pleats):
        src_x1 = int((i / num_pleats) * sw)
        src_x2 = int(((i + 1) / num_pleats) * sw)
        strip  = saree_bgr[:, max(0, src_x1):min(sw, src_x2)]
        if strip.shape[1] == 0:
            continue
        strip = cv2.resize(strip, (pleat_w_top, pleat_h))

        # Stronger alternating shadow/highlight for visible fold depth
        shade = int(80 * abs(np.sin((i + 0.5) * np.pi / num_pleats)))
        if i % 2 == 1:
            strip = np.clip(strip.astype(int) - shade, 0, 255).astype(np.uint8)
        else:
            strip = np.clip(strip.astype(int) + shade // 3, 0, 255).astype(np.uint8)

        # Trapezoid: top wider, bottom narrower — realistic fold shape
        tx     = x_start + i * pleat_w_top
        offset = (pleat_w_top - pleat_w_bot) // 2
        dst_pts = np.float32([
            [tx,                       wy],
            [tx + pleat_w_top,         wy],
            [tx + pleat_w_top - offset, wy + pleat_h],
            [tx + offset,              wy + pleat_h],
        ])
        # Add drop shadow on first pleat only to avoid overdark
        canvas = _warp_quad(strip, dst_pts, canvas, alpha=pleat_alpha,
                            shadow=(i == 0))

    return canvas


def _diagonal_pallu(canvas: np.ndarray, saree_bgr: np.ndarray,
                    wx: float, wy: float,
                    lsx: float, lsy: float,
                    sh_width: float, body_h: float,
                    alpha: float = 0.90) -> np.ndarray:
    """
    Draw pallu diagonally from right-hip up and over the left shoulder.
    Includes: diagonal body, shoulder drape, and hanging tail.
    """
    src_h, src_w = saree_bgr.shape[:2]

    # ── Diagonal body of pallu (hip → shoulder) ────────────────────────────────
    pallu_w = int(sh_width * 1.35)   # wider than before
    pallu_h = int(body_h  * 0.65)   # taller
    pallu_src = saree_bgr[:max(src_h // 2, 1), :]
    pallu_src = cv2.resize(pallu_src, (pallu_w, pallu_h))

    # Quad: right side of waist → across chest → left shoulder
    # Adjusted so top-left corner lands exactly on shoulder point
    dst_pts = np.float32([
        [lsx - sh_width * 0.15, lsy - body_h * 0.02],  # top-left  = shoulder
        [wx  + sh_width * 0.55, wy - body_h * 0.06],   # top-right = waist right
        [wx  + sh_width * 0.60, wy + body_h * 0.04],   # bot-right
        [lsx + sh_width * 0.30, lsy + body_h * 0.28],  # bot-left  = chest
    ])
    canvas = _warp_quad(pallu_src, dst_pts, canvas, alpha=alpha, shadow=True)

    # ── Shoulder drape — fabric pooling over the shoulder cap ─────────────────
    drape_w = int(sh_width * 0.65)
    drape_h = int(body_h  * 0.18)
    drape   = cv2.resize(saree_bgr[: src_h // 3, :], (drape_w, drape_h))
    # Slight perspective tilt toward body
    drap_pts = np.float32([
        [lsx - drape_w * 0.45, lsy - drape_h * 0.20],
        [lsx + drape_w * 0.35, lsy - drape_h * 0.10],
        [lsx + drape_w * 0.30, lsy + drape_h * 0.90],
        [lsx - drape_w * 0.50, lsy + drape_h * 0.80],
    ])
    canvas = _warp_quad(drape, drap_pts, canvas, alpha=alpha * 0.92)

    # ── Hanging tail from shoulder downward ───────────────────────────────────
    tail_h  = int(body_h * 0.55)   # longer tail (was 0.38)
    tail_w  = int(sh_width * 0.90) # wider tail  (was 0.80)
    tail    = cv2.resize(saree_bgr, (tail_w, tail_h))
    # Slight rotation for natural hang
    centre  = (tail_w // 2, tail_h // 2)
    rot_mat = cv2.getRotationMatrix2D(centre, 8, 1.0)
    tail    = cv2.warpAffine(tail, rot_mat, (tail_w, tail_h),
                              borderMode=cv2.BORDER_REPLICATE)
    tail_x  = int(lsx - tail_w * 0.52)
    tail_y  = int(lsy + drape_h * 0.5)
    canvas  = _overlay_image(canvas, tail, tail_x, tail_y, alpha=alpha * 0.87)

    return canvas


def _saree_lower_body(canvas: np.ndarray, saree_bgr: np.ndarray,
                      wx: float, wy: float,
                      lhx: float, lhy: float,
                      rhx: float, rhy: float,
                      lankx: float, lanky: float,
                      rankx: float, ranky: float,
                      sh_width: float,
                      alpha: float = 0.88) -> np.ndarray:
    """
    Warp the main saree body onto the lower-body trapezoid.
    Expanded padding gives a fuller, more natural skirt drape.
    """
    pad = sh_width * 0.22   # larger pad for wider, fuller coverage (was 0.15)
    dst_pts = np.float32([
        [lhx - pad,   lhy],      # top-left  (left hip)
        [rhx + pad,   rhy],      # top-right (right hip)
        [rankx + pad, ranky],    # bottom-right (right ankle)
        [lankx - pad, lanky],    # bottom-left  (left ankle)
    ])
    canvas = _warp_quad(saree_bgr, dst_pts, canvas, alpha=alpha, shadow=True)
    return canvas


# ── Style Implementations ──────────────────────────────────────────────────────

def _drape_nivi(body_bgra: np.ndarray, saree_bgr: np.ndarray, anchors: dict) -> np.ndarray:
    """Nivi style: saree wraps body, pleats at center front, pallu over left shoulder."""
    h, w   = body_bgra.shape[:2]
    result = body_bgra.copy()

    lhx,  lhy  = _a(anchors, "left_hip",        (w*0.38, h*0.52))
    rhx,  rhy  = _a(anchors, "right_hip",        (w*0.62, h*0.52))
    lankx,lanky= _a(anchors, "left_ankle",        (w*0.40, h*0.90))
    rankx,ranky= _a(anchors, "right_ankle",       (w*0.60, h*0.90))
    wx,   wy   = _a(anchors, "waist_center",      (w*0.50, h*0.50))
    lsx,  lsy  = _a(anchors, "left_shoulder",     (w*0.32, h*0.22))
    sh_w       = float(anchors.get("shoulder_width", w*0.30))
    body_h     = float(anchors.get("body_height",    h*0.70))

    # 1. Lower body saree (hip → ankle trapezoid) — main skirt portion
    result = _saree_lower_body(result, saree_bgr,
                                wx, wy, lhx, lhy, rhx, rhy,
                                lankx, lanky, rankx, ranky, sh_w, alpha=0.88)

    # 2. Fan pleats at front waist
    result = _fan_pleats(result, saree_bgr,
                         wx - sh_w * 0.05, wy, sh_w * 0.80, body_h,
                         num_pleats=7, pleat_alpha=0.85)

    # 3. Pallu: over left shoulder
    result = _diagonal_pallu(result, saree_bgr, wx, wy, lsx, lsy,
                              sh_w, body_h * 0.90, alpha=0.88)

    return result


def _drape_bridal(body_bgra: np.ndarray, saree_bgr: np.ndarray, anchors: dict) -> np.ndarray:
    """Bridal style: fuller coverage, denser pleats, wide pallu over left shoulder."""
    h, w   = body_bgra.shape[:2]
    result = body_bgra.copy()

    lhx,  lhy  = _a(anchors, "left_hip",        (w*0.36, h*0.52))
    rhx,  rhy  = _a(anchors, "right_hip",        (w*0.64, h*0.52))
    lankx,lanky= _a(anchors, "left_ankle",        (w*0.38, h*0.90))
    rankx,ranky= _a(anchors, "right_ankle",       (w*0.62, h*0.90))
    wx,   wy   = _a(anchors, "waist_center",      (w*0.50, h*0.50))
    lsx,  lsy  = _a(anchors, "left_shoulder",     (w*0.30, h*0.22))
    sh_w       = float(anchors.get("shoulder_width", w*0.32))
    body_h     = float(anchors.get("body_height",    h*0.70))

    result = _saree_lower_body(result, saree_bgr,
                                wx, wy, lhx, lhy, rhx, rhy,
                                lankx, lanky, rankx, ranky, sh_w * 1.15, alpha=0.90)
    result = _fan_pleats(result, saree_bgr, wx, wy, sh_w * 0.95, body_h, num_pleats=9)
    result = _diagonal_pallu(result, saree_bgr, wx, wy, lsx, lsy, sh_w * 1.2, body_h, alpha=0.90)
    return result


def _drape_hanging(body_bgra: np.ndarray, saree_bgr: np.ndarray, anchors: dict) -> np.ndarray:
    """Hanging style: saree sits lower on hips, looser pleats."""
    h, w   = body_bgra.shape[:2]
    result = body_bgra.copy()

    lhx,  lhy  = _a(anchors, "left_hip",        (w*0.38, h*0.55))
    rhx,  rhy  = _a(anchors, "right_hip",        (w*0.62, h*0.55))
    lankx,lanky= _a(anchors, "left_ankle",        (w*0.40, h*0.92))
    rankx,ranky= _a(anchors, "right_ankle",       (w*0.60, h*0.92))
    wx,   wy   = _a(anchors, "waist_center",      (w*0.50, h*0.55))
    lsx,  lsy  = _a(anchors, "left_shoulder",     (w*0.32, h*0.22))
    sh_w       = float(anchors.get("shoulder_width", w*0.30))
    body_h     = float(anchors.get("body_height",    h*0.65))

    result = _saree_lower_body(result, saree_bgr,
                                wx, wy, lhx, lhy, rhx, rhy,
                                lankx, lanky, rankx, ranky, sh_w, alpha=0.86)
    result = _fan_pleats(result, saree_bgr, wx, wy, sh_w * 0.78, body_h, num_pleats=5)
    result = _diagonal_pallu(result, saree_bgr, wx, wy, lsx, lsy, sh_w, body_h * 0.88, alpha=0.85)
    return result


def _drape_gujarati(body_bgra: np.ndarray, saree_bgr: np.ndarray, anchors: dict) -> np.ndarray:
    """Gujarati style: pallu pinned at front right shoulder."""
    h, w   = body_bgra.shape[:2]
    result = body_bgra.copy()

    lhx,  lhy  = _a(anchors, "left_hip",        (w*0.38, h*0.52))
    rhx,  rhy  = _a(anchors, "right_hip",        (w*0.62, h*0.52))
    lankx,lanky= _a(anchors, "left_ankle",        (w*0.40, h*0.90))
    rankx,ranky= _a(anchors, "right_ankle",       (w*0.60, h*0.90))
    wx,   wy   = _a(anchors, "waist_center",      (w*0.50, h*0.50))
    rsx,  rsy  = _a(anchors, "right_shoulder",    (w*0.68, h*0.22))
    sh_w       = float(anchors.get("shoulder_width", w*0.30))
    body_h     = float(anchors.get("body_height",    h*0.70))

    result = _saree_lower_body(result, saree_bgr,
                                wx, wy, lhx, lhy, rhx, rhy,
                                lankx, lanky, rankx, ranky, sh_w, alpha=0.88)
    result = _fan_pleats(result, saree_bgr, wx, wy, sh_w * 0.80, body_h, num_pleats=7)
    # Gujarati: pallu goes over RIGHT shoulder
    result = _diagonal_pallu(result, saree_bgr, wx, wy, rsx, rsy, sh_w, body_h, alpha=0.88)
    return result


# ── Pleat & Pallu (legacy stubs kept for backward compat) ─────────────────────

def _add_pleats(canvas, saree_bgr, waist, width, height, num_pleats=7):
    return _fan_pleats(canvas, saree_bgr,
                       float(waist["x"]), float(waist["y"]),
                       float(width), float(height), num_pleats)


def _add_pallu(canvas, saree_bgr, shoulder, width, height):
    h, w = canvas.shape[:2]
    wx   = float(shoulder["x"]) + float(width) * 0.6
    wy   = float(shoulder["y"]) + float(height) * 0.2
    return _diagonal_pallu(canvas, saree_bgr,
                           wx, wy,
                           float(shoulder["x"]), float(shoulder["y"]),
                           float(width), float(height))


# ── Fabric Simulation ──────────────────────────────────────────────────────────

def _simulate_fabric(image: np.ndarray, fabric_type: str) -> np.ndarray:
    """
    Apply fabric-specific visual effects to the composite image.
    """
    if fabric_type == "silk":
        # Add specular highlight using overlay blend
        image = _add_silk_sheen(image)
    elif fabric_type in ("chiffon", "georgette"):
        # Slight transparency / soft blur
        if image.shape[2] == 4:
            bgr, alpha = image[:, :, :3], image[:, :, 3]
            bgr   = cv2.GaussianBlur(bgr, (3, 3), 0)
            image = np.dstack([bgr, alpha])
        else:
            image = cv2.GaussianBlur(image, (3, 3), 0)
    elif fabric_type == "cotton":
        # Slight texture / matte effect
        noise = np.random.randint(-8, 8, image.shape[:2] + (3,), dtype=np.int16)
        if image.shape[2] == 4:
            bgr, alpha = image[:, :, :3].astype(np.int16) + noise, image[:, :, 3]
            image = np.dstack([np.clip(bgr, 0, 255).astype(np.uint8), alpha])
        else:
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def _add_silk_sheen(image: np.ndarray) -> np.ndarray:
    """Add a diagonal highlight streak to simulate silk sheen."""
    h, w = image.shape[:2]
    sheen = np.zeros((h, w), dtype=np.uint8)
    cv2.line(sheen, (0, 0), (w, h), 40, max(w // 6, 20))
    sheen = cv2.GaussianBlur(sheen, (51, 51), 0)
    if image.shape[2] == 4:
        bgr, alpha = image[:, :, :3], image[:, :, 3]
        for c in range(3):
            bgr[:, :, c] = np.clip(bgr[:, :, c].astype(int) + sheen, 0, 255)
        return np.dstack([bgr, alpha])
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c].astype(int) + sheen, 0, 255)
    return image


# ── Composite Overlay Helper ───────────────────────────────────────────────────

def _overlay_image(
    background: np.ndarray,
    overlay_bgr: np.ndarray,
    x: int,
    y: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Paste overlay_bgr (BGR or BGRA) onto background (BGR or BGRA) at (x, y).
    Handles boundary clipping.
    """
    bh, bw = background.shape[:2]
    oh, ow = overlay_bgr.shape[:2]

    # Clip overlay to canvas bounds
    x1b, y1b = max(x, 0), max(y, 0)
    x2b, y2b = min(x + ow, bw), min(y + oh, bh)
    x1o = x1b - x
    y1o = y1b - y
    x2o = x1o + (x2b - x1b)
    y2o = y1o + (y2b - y1b)

    if x2b <= x1b or y2b <= y1b:
        return background

    roi     = background[y1b:y2b, x1b:x2b]
    overlay = overlay_bgr[y1o:y2o, x1o:x2o]

    if overlay_bgr.shape[2] == 4:
        ov_alpha = (overlay[:, :, 3] / 255.0 * alpha)[:, :, np.newaxis]
        ov_bgr   = overlay[:, :, :3].astype(float)
    else:
        ov_alpha = np.full((overlay.shape[0], overlay.shape[1], 1), alpha)
        ov_bgr   = overlay.astype(float)

    bg_bgr = roi[:, :, :3].astype(float)
    blended = (ov_bgr * ov_alpha + bg_bgr * (1 - ov_alpha)).astype(np.uint8)

    result = background.copy()
    if result.shape[2] == 4:
        result[y1b:y2b, x1b:x2b, :3] = blended
    else:
        result[y1b:y2b, x1b:x2b] = blended

    return result


# Bind style functions to dict (defined after functions exist)
DRAPING_STYLES = {
    "nivi":     _drape_nivi,
    "bridal":   _drape_bridal,
    "hanging":  _drape_hanging,
    "gujarati": _drape_gujarati,
}
