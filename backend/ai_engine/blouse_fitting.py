"""
Blouse Fitting Module
=====================
Fits the blouse material onto the body image using:
 - Body keypoints from pose detection
 - Geometric transformation (affine / perspective warp)
 - Neck & sleeve masking based on customization options
"""

import cv2
import numpy as np
import os
from typing import Optional


# Neck cutout templates as polygon fractions of the blouse bounding box
# Values are (x_frac, y_frac) normalised to blouse width/height
NECK_TEMPLATES = {
    "round":       [(0.35, 0.0), (0.65, 0.0), (0.65, 0.22), (0.5, 0.28), (0.35, 0.22)],
    "boat":        [(0.20, 0.0), (0.80, 0.0), (0.75, 0.18), (0.25, 0.18)],
    "v_neck":      [(0.40, 0.0), (0.60, 0.0), (0.50, 0.35)],
    "square":      [(0.30, 0.0), (0.70, 0.0), (0.70, 0.22), (0.30, 0.22)],
    "sweetheart":  [(0.30, 0.0), (0.70, 0.0), (0.72, 0.20), (0.60, 0.28),
                    (0.50, 0.24), (0.40, 0.28), (0.28, 0.20)],
}

# Sleeve length as a fraction of arm length (shoulder→wrist)
SLEEVE_FRACTIONS = {
    "sleeveless": 0.0,
    "short":      0.25,
    "half":       0.5,
    "full":       1.0,
}


def fit_blouse(
    body_image_path: str,
    blouse_image_path: str,
    pose_data: dict,
    neck_type: str = "round",
    sleeve_type: str = "short",
    color: str = "#FFFFFF",
) -> dict:
    """
    Fit the blouse onto the body and return the path of the composited image.

    Returns:
        dict with keys:
            blouse_overlay_path – path to blouse-only BGRA overlay
            shoulder_width      – detected shoulder width in pixels
            blouse_height       – computed blouse height in pixels
    """
    if not os.path.exists(body_image_path):
        raise FileNotFoundError(f"Body image not found: {body_image_path}")
    if not os.path.exists(blouse_image_path):
        raise FileNotFoundError(f"Blouse image not found: {blouse_image_path}")

    body_bgr   = cv2.imread(body_image_path)
    blouse_bgr = cv2.imread(blouse_image_path)
    h, w       = body_bgr.shape[:2]

    # Scale anchors from original detection size to current body image size
    raw_anchors = pose_data.get("draping_anchors", {})
    orig_size   = pose_data.get("image_size", {})
    orig_w      = orig_size.get("width",  w)
    orig_h      = orig_size.get("height", h)
    sx, sy      = w / orig_w, h / orig_h

    anchors = {}
    for k, v in raw_anchors.items():
        if isinstance(v, dict) and "x" in v:
            anchors[k] = {"x": v["x"] * sx, "y": v["y"] * sy}
        elif isinstance(v, (int, float)):
            anchors[k] = v * (sx if "width" in k else sy)
        else:
            anchors[k] = v

    # ── Compute blouse bounding box ────────────────────────────────────────────
    shoulder_c = anchors.get("shoulder_center", {"x": w / 2, "y": h * 0.2})
    waist_c    = anchors.get("waist_center",    {"x": w / 2, "y": h * 0.5})
    sh_width   = anchors.get("shoulder_width",  w * 0.3)
    torso_h    = anchors.get("torso_height",    h * 0.3)

    blouse_w = int(sh_width * 1.60)   # wider: was 1.4
    blouse_h = int(torso_h  * 0.90)   # taller: was 0.65

    # ── Colorise blouse material ───────────────────────────────────────────────
    blouse_colored = _colorise(blouse_bgr, color)

    sc_x    = float(shoulder_c["x"] if isinstance(shoulder_c, dict) else w / 2)
    sc_y    = float(shoulder_c["y"] if isinstance(shoulder_c, dict) else h * 0.2)
    wc_x    = float(waist_c["x"]    if isinstance(waist_c,    dict) else w / 2)
    wc_y    = float(waist_c["y"]    if isinstance(waist_c,    dict) else h * 0.5)
    half_sh = sh_width / 2

    # ── Warp blouse texture onto torso quad → produces full-canvas BGRA ──────
    # Source rect -> shoulder/waist trapezoid on the body
    tq_src = np.float32([[0, 0], [blouse_w, 0], [blouse_w, blouse_h], [0, blouse_h]])
    tq_dst = np.float32([
        [sc_x - half_sh * 0.80, sc_y],           # top-left  (left shoulder)
        [sc_x + half_sh * 0.80, sc_y],           # top-right (right shoulder)
        [wc_x + half_sh * 0.55, wc_y],           # bot-right (right waist)
        [wc_x - half_sh * 0.55, wc_y],           # bot-left  (left waist)
    ])
    M_torso = cv2.getPerspectiveTransform(tq_src, tq_dst)

    # Warp onto a transparent BGRA canvas the same size as the body image
    blouse_tile = cv2.resize(blouse_colored, (blouse_w, blouse_h))
    blouse_bgra_full = cv2.cvtColor(blouse_tile, cv2.COLOR_BGR2BGRA)
    # Warp the colour canvas
    warped_bgr = cv2.warpPerspective(
        blouse_tile, M_torso, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    # Warp a white mask to know where fabric landed
    mask_src = np.ones((blouse_h, blouse_w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask_src, M_torso, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # Feather mask edges
    warped_mask = cv2.GaussianBlur(warped_mask, (9, 9), 0)

    # ── Apply neck cutout to the warp mask ───────────────────────────────────
    warped_mask = _apply_neck_cutout_to_mask(warped_mask, neck_type,
                                              tq_dst, sc_x, sc_y, wc_x, wc_y, w, h)

    # ── Build full-canvas BGRA overlay (no black fill outside blouse) ─────────
    blouse_canvas = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2BGRA)
    blouse_canvas[:, :, 3] = warped_mask  # alpha = warped mask only

    # ── Add sleeves on the canvas ────────────────────────────────────────────
    blouse_canvas = _apply_sleeves_canvas(
        blouse_canvas, blouse_colored, anchors, sleeve_type,
        sc_x, sc_y, half_sh, w, h,
    )

    # ── Save blouse overlay ──────────────────────────────────────────────────
    import uuid
    overlay_dir  = os.path.join(os.path.dirname(__file__), "..", "outputs", "blouse_overlays")
    overlay_dir  = os.path.normpath(overlay_dir)
    os.makedirs(overlay_dir, exist_ok=True)
    overlay_path = os.path.join(overlay_dir, f"{uuid.uuid4().hex}.png")
    cv2.imwrite(overlay_path, blouse_canvas)

    print(f"[Blouse] sc=({sc_x:.0f},{sc_y:.0f}) wc=({wc_x:.0f},{wc_y:.0f}) "
          f"sh_w={sh_width:.0f} blouse={blouse_w}x{blouse_h} overlay={overlay_path}")

    return {
        "blouse_overlay_path": overlay_path,
        "shoulder_width":      sh_width,
        "blouse_height":       blouse_h,
        "position":            {"x": 0, "y": 0},  # canvas is full-size
    }


def _colorise(bgr: np.ndarray, hex_color: str) -> np.ndarray:
    """
    Tint blouse with chosen colour using HSV hue+saturation replacement.
    Preserves the original luminance (texture detail) while applying colour.
    """
    hex_color = hex_color.lstrip("#")
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        return bgr

    # Convert target colour to HSV to get hue & saturation
    target_bgr = np.uint8([[[b, g, r]]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    t_hue = int(target_hsv[0])
    t_sat = int(target_hsv[1])

    # Convert image to HSV, replace hue & saturation, keep value
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = t_hue
    hsv[:, :, 1] = np.clip(t_sat * 0.8 + hsv[:, :, 1] * 0.2, 0, 255)
    # Keep Value channel (brightness/texture) unchanged
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def _apply_neck_cutout_to_mask(
    mask: np.ndarray,
    neck_type: str,
    tq_dst: np.ndarray,
    sc_x: float, sc_y: float,
    wc_x: float, wc_y: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """
    Carve a neck hole into the full-canvas warp mask.
    Neck polygon is defined relative to the top of the blouse trapezoid.
    """
    points = NECK_TEMPLATES.get(neck_type, NECK_TEMPLATES["round"])

    # Compute bounding box of the blouse trapezoid top edge
    top_left_x  = float(tq_dst[0][0])
    top_right_x = float(tq_dst[1][0])
    blouse_top_y = float(tq_dst[0][1])
    blouse_bot_y = float(tq_dst[3][1])
    blouse_w_px  = top_right_x - top_left_x
    blouse_h_px  = blouse_bot_y - blouse_top_y

    poly = np.array(
        [
            (int(top_left_x + x * blouse_w_px),
             int(blouse_top_y + y * blouse_h_px))
            for x, y in points
        ],
        dtype=np.int32,
    )
    result = mask.copy()
    cv2.fillPoly(result, [poly], 0)
    result = cv2.GaussianBlur(result, (9, 9), 0)
    return result


def _apply_sleeves_canvas(
    blouse_canvas: np.ndarray,
    blouse_colored: np.ndarray,
    anchors: dict,
    sleeve_type: str,
    sc_x: float, sc_y: float,
    half_sh: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """
    Draw sleeves as filled trapezoid polygons directly on the full-canvas
    BGRA blouse overlay. Each sleeve is a narrow quad from shoulder to
    end-point. Alpha is set so only the sleeve polygon area is painted.
    """
    sleeve_frac = SLEEVE_FRACTIONS.get(sleeve_type, 0.25)
    if sleeve_frac == 0.0:
        return blouse_canvas

    left_sh  = anchors.get("left_shoulder",  {"x": img_w * 0.30, "y": img_h * 0.22})
    right_sh = anchors.get("right_shoulder", {"x": img_w * 0.70, "y": img_h * 0.22})
    left_wr  = anchors.get("left_wrist",     {"x": img_w * 0.20, "y": img_h * 0.55})
    right_wr = anchors.get("right_wrist",    {"x": img_w * 0.80, "y": img_h * 0.55})

    sleeve_color = _dominant_color_bgr(blouse_colored)
    sleeve_half  = max(int(half_sh * 0.28), 14)   # half-width of sleeve
    result = blouse_canvas.copy()

    for sh, wr in [(left_sh, left_wr), (right_sh, right_wr)]:
        shx, shy = float(sh["x"]), float(sh["y"])
        wrx = shx + sleeve_frac * (float(wr["x"]) - shx)
        wry = shy + sleeve_frac * (float(wr["y"]) - shy)

        # Direction perpendicular to arm for sleeve width
        dx, dy = wrx - shx, wry - shy
        length  = max(np.hypot(dx, dy), 1.0)
        px, py  = -dy / length * sleeve_half, dx / length * sleeve_half

        poly = np.array([
            [int(shx + px), int(shy + py)],
            [int(shx - px), int(shy - py)],
            [int(wrx - px * 0.7), int(wry - py * 0.7)],
            [int(wrx + px * 0.7), int(wry + py * 0.7)],
        ], dtype=np.int32)

        # Draw on BGR channels
        bgr_layer = result[:, :, :3].copy()
        cv2.fillPoly(bgr_layer, [poly], sleeve_color)

        # Draw on alpha channel (opaque where sleeve is)
        alpha_layer = result[:, :, 3].copy()
        cv2.fillPoly(alpha_layer, [poly], 220)
        alpha_layer = cv2.GaussianBlur(alpha_layer, (5, 5), 0)

        result[:, :, :3] = bgr_layer
        result[:, :, 3]  = alpha_layer

    return result


def _dominant_color_bgr(bgr: np.ndarray) -> tuple:
    """Return the dominant BGR colour of an image as a tuple."""
    small  = cv2.resize(bgr, (20, 20)).reshape(-1, 3)
    avg    = small.mean(axis=0).astype(int)
    return (int(avg[0]), int(avg[1]), int(avg[2]))
