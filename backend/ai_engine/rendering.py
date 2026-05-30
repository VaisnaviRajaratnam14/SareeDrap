"""
Rendering Module
================
Orchestrates the full draping pipeline:
  1. Load and segment body image
  2. Drape saree using selected style
  3. Overlay fitted blouse
  4. Post-process (sharpening, colour grading)
  5. Save final output image
"""

import cv2
import numpy as np
import os
import uuid



def render_draped_image(
    body_image_path: str,
    saree_image_path: str,
    blouse_image_path: str,
    pose_data: dict,
    saree_features: dict,
    blouse_config: dict,
    draping_style: str,
    output_folder: str,
) -> str:
    """
    Full pipeline: returns path to final rendered output image.

    Args:
        body_image_path:   Path to uploaded body image.
        saree_image_path:  Path to uploaded saree image.
        blouse_image_path: Path to uploaded blouse image.
        pose_data:         Output from pose_detection.detect_pose().
        saree_features:    Output from saree_draping.extract_saree_features().
        blouse_config:     { neck_type, sleeve_type, color }.
        draping_style:     nivi | bridal | hanging | gujarati.
        output_folder:     Directory to save the output image.

    Returns:
        Absolute path to the saved output image.
    """
    # ── Step 1: Load images ────────────────────────────────────────────────────
    body_bgr  = cv2.imread(body_image_path)
    saree_bgr = cv2.imread(saree_image_path)

    if body_bgr is None:
        raise ValueError(f"Cannot read body image: {body_image_path}")
    if saree_bgr is None:
        raise ValueError(f"Cannot read saree image: {saree_image_path}")

    # ── Lazy imports ───────────────────────────────────────────────────────────
    from ai_engine.segmentation import get_body_mask, apply_mask_to_image
    from ai_engine.saree_draping import drape_saree
    from ai_engine.blouse_fitting import fit_blouse

    # ── Step 2: Get body mask BEFORE preprocessing (at original resolution) ────
    body_mask_orig = get_body_mask(body_image_path)

    # ── Step 3: Preprocess body (resize + enhance) ─────────────────────────────
    body_bgr = _preprocess_image(body_bgr)
    h, w     = body_bgr.shape[:2]

    # Resize mask to match preprocessed dimensions
    body_mask = cv2.resize(body_mask_orig, (w, h), interpolation=cv2.INTER_NEAREST)
    body_bgra = apply_mask_to_image(body_bgr, body_mask)

    # ── Step 4: Drape saree ────────────────────────────────────────────────────
    raw_anchors = pose_data.get("draping_anchors", _default_anchors(w, h))
    orig_size   = pose_data.get("image_size", {})
    anchors     = _scale_anchors(raw_anchors, orig_size, w, h)
    fabric_type = saree_features.get("fabric_type", "silk")
    draped_bgra = drape_saree(body_bgra, saree_bgr, anchors, draping_style, fabric_type)

    # Re-apply softened body mask so saree blends naturally within silhouette
    expanded_mask = _expand_mask(body_mask, expand_px=22)
    if draped_bgra.shape[2] == 4:
        orig_alpha = draped_bgra[:, :, 3].astype(np.float32)
        clip_alpha = expanded_mask.astype(np.float32)
        draped_bgra[:, :, 3] = np.minimum(orig_alpha, clip_alpha).astype(np.uint8)
    else:
        draped_bgra = cv2.cvtColor(draped_bgra, cv2.COLOR_BGR2BGRA)
        draped_bgra[:, :, 3] = expanded_mask

    # ── Step 5: Fit blouse ─────────────────────────────────────────────────────
    neck_type   = blouse_config.get("neck_type",   "round")
    sleeve_type = blouse_config.get("sleeve_type", "short")
    color       = blouse_config.get("color",       "#FFFFFF")

    print(f"[Rendering] body={body_image_path} saree={saree_image_path} blouse={blouse_image_path}")
    print(f"[Rendering] body_size={w}x{h} style={draping_style} fabric={fabric_type}")
    print(f"[Rendering] anchors: shoulder_center={raw_anchors.get('shoulder_center')} "
          f"waist_center={raw_anchors.get('waist_center')} "
          f"shoulder_width={raw_anchors.get('shoulder_width')}")

    try:
        blouse_result = fit_blouse(
            body_image_path=body_image_path,
            blouse_image_path=blouse_image_path,
            pose_data=pose_data,
            neck_type=neck_type,
            sleeve_type=sleeve_type,
            color=color,
        )
        blouse_overlay_path = blouse_result["blouse_overlay_path"]
        blouse_position     = blouse_result["position"]
        print(f"[Rendering] blouse overlay: {blouse_overlay_path} pos={blouse_position}")

        blouse_overlay = cv2.imread(blouse_overlay_path, cv2.IMREAD_UNCHANGED)
        if blouse_overlay is not None:
            # Resize blouse canvas to match current body size if needed
            if blouse_overlay.shape[:2] != (h, w):
                blouse_overlay = cv2.resize(blouse_overlay, (w, h), interpolation=cv2.INTER_LINEAR)

            # Clip blouse alpha to body mask so it never paints outside the silhouette
            if blouse_overlay.shape[2] == 4:
                body_mask_f   = body_mask.astype(np.float32) / 255.0
                blouse_alpha  = blouse_overlay[:, :, 3].astype(np.float32) / 255.0
                blouse_alpha  = np.minimum(blouse_alpha, body_mask_f)
                blouse_overlay[:, :, 3] = (blouse_alpha * 255).astype(np.uint8)

            draped_bgra = _composite_blouse(
                draped_bgra, blouse_overlay,
                blouse_position["x"], blouse_position["y"],
            )
    except Exception as e:
        import traceback
        print(f"[Rendering] Blouse fitting warning: {e}")
        traceback.print_exc()

    # ── Step 6: Create final composite on warm off-white background ────────────
    final = _flatten_on_background(draped_bgra, (245, 242, 238))

    # ── Step 7: Post-process ───────────────────────────────────────────────────
    final = _post_process(final)

    # ── Step 8: Save output ────────────────────────────────────────────────────
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"output_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"[Rendering] ✅ output saved: {output_path} ({final.shape[1]}x{final.shape[0]})")

    return output_path


# ── Image Preprocessing ────────────────────────────────────────────────────────

def _preprocess_image(bgr: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """
    Resize so the longer side = target_size, normalise lighting.
    """
    h, w  = bgr.shape[:2]
    scale = target_size / max(h, w)
    if scale != 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=interp)

    # CLAHE for better contrast
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l     = clahe.apply(l)
    bgr   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return bgr


# ── Blouse Compositing ─────────────────────────────────────────────────────────

def _composite_blouse(
    base_bgra: np.ndarray,
    blouse_bgra: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """Paste the blouse BGRA overlay onto the base BGRA image."""
    bh, bw = base_bgra.shape[:2]
    oh, ow = blouse_bgra.shape[:2]

    x1b, y1b = max(x, 0), max(y, 0)
    x2b, y2b = min(x + ow, bw), min(y + oh, bh)
    x1o = x1b - x
    y1o = y1b - y
    x2o = x1o + (x2b - x1b)
    y2o = y1o + (y2b - y1b)

    if x2b <= x1b or y2b <= y1b:
        return base_bgra

    roi            = base_bgra[y1b:y2b, x1b:x2b]
    blouse_crop    = blouse_bgra[y1o:y2o, x1o:x2o]

    if blouse_crop.shape[2] == 4:
        b_alpha = blouse_crop[:, :, 3:4].astype(float) / 255.0
        b_bgr   = blouse_crop[:, :, :3].astype(float)
    else:
        b_alpha = np.ones((blouse_crop.shape[0], blouse_crop.shape[1], 1))
        b_bgr   = blouse_crop.astype(float)

    base_bgr  = roi[:, :, :3].astype(float)
    blended   = (b_bgr * b_alpha + base_bgr * (1 - b_alpha)).astype(np.uint8)

    result = base_bgra.copy()
    result[y1b:y2b, x1b:x2b, :3] = blended
    return result


# ── Flatten BGRA → BGR ────────────────────────────────────────────────────────

def _flatten_on_background(
    bgra: np.ndarray, bg_color: tuple = (255, 255, 255)
) -> np.ndarray:
    """Composite BGRA over a solid background colour and return BGR."""
    bg = np.full(bgra.shape[:2] + (3,), bg_color, dtype=np.uint8)
    if bgra.shape[2] == 4:
        alpha = bgra[:, :, 3:4].astype(float) / 255.0
        fg    = bgra[:, :, :3].astype(float)
        bg_f  = bg.astype(float)
        result = (fg * alpha + bg_f * (1 - alpha)).astype(np.uint8)
    else:
        result = bgra[:, :, :3]
    return result


# ── Post-Processing ────────────────────────────────────────────────────────────

def _post_process(bgr: np.ndarray) -> np.ndarray:
    """Apply sharpening, colour enhancement, and vignette."""
    # Unsharp mask for crispness
    blurred   = cv2.GaussianBlur(bgr, (0, 0), 2.5)
    sharpened = cv2.addWeighted(bgr, 1.45, blurred, -0.45, 0)

    # Subtle saturation + warmth boost
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    sharpened = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Soft vignette: darken corners for studio-photo look
    h, w   = sharpened.shape[:2]
    cx, cy = w / 2, h / 2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vign   = np.clip(1.0 - dist * 0.30, 0.70, 1.0).astype(np.float32)
    vign   = vign[:, :, np.newaxis]
    sharpened = np.clip(sharpened.astype(np.float32) * vign, 0, 255).astype(np.uint8)

    return sharpened


# ── Default Anchors Fallback ───────────────────────────────────────────────────

def _expand_mask(mask: np.ndarray, expand_px: int = 12) -> np.ndarray:
    """Dilate and feather mask edges for soft blending."""
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    blurred = cv2.GaussianBlur(dilated, (expand_px | 1, expand_px | 1), 0)
    return blurred


def _scale_anchors(anchors: dict, orig_size: dict, new_w: int, new_h: int) -> dict:
    """Scale anchor coordinates from original image size to preprocessed size."""
    orig_w = orig_size.get("width",  new_w)
    orig_h = orig_size.get("height", new_h)
    if orig_w == new_w and orig_h == new_h:
        return anchors

    sx = new_w / orig_w
    sy = new_h / orig_h
    scaled = {}
    for key, val in anchors.items():
        if isinstance(val, dict) and "x" in val and "y" in val:
            scaled[key] = {"x": round(val["x"] * sx, 2), "y": round(val["y"] * sy, 2)}
        elif isinstance(val, (int, float)):
            # Scalar measurements: use appropriate scale factor
            if "width" in key or key in ("shoulder_width",):
                scaled[key] = round(val * sx, 2)
            else:
                scaled[key] = round(val * sy, 2)
        else:
            scaled[key] = val
    return scaled


def _default_anchors(w: int, h: int) -> dict:
    """Return sensible default anchor positions when pose detection fails."""
    return {
        "shoulder_center": {"x": w / 2, "y": h * 0.22},
        "waist_center":    {"x": w / 2, "y": h * 0.50},
        "left_shoulder":   {"x": w * 0.32, "y": h * 0.22},
        "right_shoulder":  {"x": w * 0.68, "y": h * 0.22},
        "left_hip":        {"x": w * 0.35, "y": h * 0.55},
        "right_hip":       {"x": w * 0.65, "y": h * 0.55},
        "left_ankle":      {"x": w * 0.38, "y": h * 0.92},
        "right_ankle":     {"x": w * 0.62, "y": h * 0.92},
        "left_wrist":      {"x": w * 0.18, "y": h * 0.62},
        "right_wrist":     {"x": w * 0.82, "y": h * 0.62},
        "left_elbow":      {"x": w * 0.22, "y": h * 0.42},
        "right_elbow":     {"x": w * 0.78, "y": h * 0.42},
        "shoulder_width":  w * 0.36,
        "torso_height":    h * 0.28,
        "body_height":     h * 0.70,
    }
