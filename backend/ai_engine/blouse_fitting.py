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

    # ── Warp blouse texture onto torso quad (shoulders → waist trapezoid) ──────
    sc_x = float(shoulder_c["x"] if isinstance(shoulder_c, dict) else w / 2)
    sc_y = float(shoulder_c["y"] if isinstance(shoulder_c, dict) else h * 0.2)
    wc_x = float(waist_c["x"] if isinstance(waist_c, dict) else w / 2)
    wc_y = float(waist_c["y"] if isinstance(waist_c, dict) else h * 0.5)
    half_sh = sh_width / 2
    tq_src = np.float32([[0, 0], [blouse_w, 0], [blouse_w, blouse_h], [0, blouse_h]])
    tq_dst = np.float32([
        [sc_x - half_sh * 0.82, sc_y],
        [sc_x + half_sh * 0.82, sc_y],
        [wc_x + half_sh * 0.58, wc_y],
        [wc_x - half_sh * 0.58, wc_y],
    ])
    M_torso = cv2.getPerspectiveTransform(tq_src, tq_dst)
    blouse_warped = cv2.warpPerspective(
        cv2.resize(blouse_colored, (blouse_w, blouse_h)),
        M_torso, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    bx = max(0, int(sc_x - blouse_w / 2))
    by = max(0, int(sc_y))
    ex = min(w, bx + blouse_w)
    ey = min(h, by + blouse_h)
    blouse_resized = blouse_warped[by:ey, bx:ex]
    if blouse_resized.shape[0] < 4 or blouse_resized.shape[1] < 4:
        blouse_resized = cv2.resize(blouse_colored, (blouse_w, blouse_h))

    # ── Apply neck cutout ──────────────────────────────────────────────────────
    blouse_with_neck = _apply_neck_cutout(blouse_resized, neck_type)

    # ── Apply sleeves ──────────────────────────────────────────────────────────
    blouse_with_sleeves = _apply_sleeves(
        blouse_with_neck, blouse_bgr, anchors, sleeve_type, blouse_w, blouse_h, w, h
    )

    # ── Save blouse overlay ────────────────────────────────────────────────────
    import uuid
    overlay_dir  = os.path.join(os.path.dirname(__file__), "..", "outputs", "blouse_overlays")
    overlay_dir  = os.path.normpath(overlay_dir)
    os.makedirs(overlay_dir, exist_ok=True)
    overlay_path = os.path.join(overlay_dir, f"{uuid.uuid4().hex}.png")
    cv2.imwrite(overlay_path, blouse_with_sleeves)

    return {
        "blouse_overlay_path": overlay_path,
        "shoulder_width":      sh_width,
        "blouse_height":       blouse_h,
        "position": {
            "x": int(shoulder_c["x"] - blouse_w / 2),
            "y": int(shoulder_c["y"]),
        },
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


def _apply_neck_cutout(blouse_bgr: np.ndarray, neck_type: str) -> np.ndarray:
    """
    Cut a neck hole in the blouse using a polygon mask with feathered edges.
    Returns BGRA with transparent neck area.
    """
    h, w   = blouse_bgr.shape[:2]
    points = NECK_TEMPLATES.get(neck_type, NECK_TEMPLATES["round"])

    poly = np.array(
        [(int(x * w), int(y * h)) for x, y in points],
        dtype=np.int32,
    )

    # Fully opaque base, zero out neck, then feather edge with blur
    alpha = np.full((h, w), 255, dtype=np.uint8)
    cv2.fillPoly(alpha, [poly], 0)
    # Feather the cutout edge for natural look
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    bgra = cv2.cvtColor(blouse_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def _apply_sleeves(
    blouse_bgra: np.ndarray,
    blouse_texture: np.ndarray,
    anchors: dict,
    sleeve_type: str,
    blouse_w: int,
    blouse_h: int,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Draw left and right sleeves on the blouse overlay based on sleeve_type.
    """
    sleeve_frac = SLEEVE_FRACTIONS.get(sleeve_type, 0.25)
    if sleeve_frac == 0.0:
        return blouse_bgra

    # Arm vectors from shoulder to elbow / wrist
    left_sh  = anchors.get("left_shoulder",  {"x": img_w * 0.3,  "y": img_h * 0.2})
    right_sh = anchors.get("right_shoulder", {"x": img_w * 0.7,  "y": img_h * 0.2})
    left_el  = anchors.get("left_elbow",     {"x": img_w * 0.25, "y": img_h * 0.4})
    right_el = anchors.get("right_elbow",    {"x": img_w * 0.75, "y": img_h * 0.4})
    left_wr  = anchors.get("left_wrist",     {"x": img_w * 0.2,  "y": img_h * 0.6})
    right_wr = anchors.get("right_wrist",    {"x": img_w * 0.8,  "y": img_h * 0.6})

    def arm_end(shoulder, elbow, wrist, frac):
        """Interpolate endpoint along shoulder → wrist at given fraction."""
        ex = shoulder["x"] + frac * (wrist["x"] - shoulder["x"])
        ey = shoulder["y"] + frac * (wrist["y"] - shoulder["y"])
        return {"x": ex, "y": ey}

    left_end  = arm_end(left_sh,  left_el,  left_wr,  sleeve_frac)
    right_end = arm_end(right_sh, right_el, right_wr, sleeve_frac)

    sleeve_width = max(int(blouse_w * 0.22), 20)

    # Draw sleeve rectangles on the blouse canvas (in blouse-local coords)
    # Convert body coords → blouse-local coords
    bx0 = right_sh["x"] - blouse_w / 2
    by0 = right_sh["y"]

    def to_local(pt):
        return (int(pt["x"] - bx0), int(pt["y"] - by0))

    canvas = blouse_bgra.copy()
    bh, bw = canvas.shape[:2]

    for sh, end in [(left_sh, left_end), (right_sh, right_end)]:
        p1 = to_local(sh)
        p2 = to_local(end)
        if 0 <= p1[0] < bw and 0 <= p1[1] < bh:
            # Draw filled rectangle representing sleeve
            cv2.rectangle(
                canvas,
                (p1[0] - sleeve_width // 2, p1[1]),
                (p2[0] + sleeve_width // 2, p2[1]),
                _dominant_color_bgr(blouse_texture),
                thickness=-1,
            )

    return canvas


def _dominant_color_bgr(bgr: np.ndarray) -> tuple:
    """Return the dominant BGR colour of an image as a tuple."""
    small  = cv2.resize(bgr, (20, 20)).reshape(-1, 3)
    avg    = small.mean(axis=0).astype(int)
    return (int(avg[0]), int(avg[1]), int(avg[2]))
