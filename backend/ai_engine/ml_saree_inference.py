"""
ML Saree Try-On Inference
==========================
Loads trained SareeTryOnModel weights and runs the full inference pipeline.
Falls back to the geometric engine if weights are absent.

Weight path: backend/dataset/trained_models/saree_tryon_model.pth
"""

import os
import cv2
import json
import uuid
import numpy as np
from pathlib import Path
from typing import Optional

# ── Weight path ────────────────────────────────────────────────────────────────
_WEIGHTS_PATH = Path(__file__).parent.parent / "dataset" / "trained_models" / "saree_tryon_model.pth"
_CONFIG_PATH  = Path(__file__).parent.parent / "dataset" / "trained_models" / "model_config.json"

# ── Singleton model cache ──────────────────────────────────────────────────────
_model  = None
_device = None
_config = None

# Default model config — overridden by model_config.json if present
_DEFAULT_CONFIG = {
    "img_h":   256,
    "img_w":   192,
    "grid_size": 5,
    "base_ch":  64,
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std":  [0.5, 0.5, 0.5],
}

# Pose keypoint names matching the 33-landmark MediaPipe output
_HEATMAP_KPS = [
    "left_shoulder", "right_shoulder", "left_elbow",  "right_elbow",
    "left_wrist",    "right_wrist",    "left_hip",    "right_hip",
    "left_knee",     "right_knee",     "left_ankle",  "right_ankle",
    "nose",          "left_ear",       "right_ear",
    "left_eye",      "right_eye",      "mouth_left",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def weights_available() -> bool:
    """Return True if trained model weights FILE exists and is >1 KB."""
    return _WEIGHTS_PATH.exists() and _WEIGHTS_PATH.stat().st_size > 1024 \
        if _WEIGHTS_PATH.exists() else False


def verify_weights() -> dict:
    """
    Deep-verify the weights file:
    1. Check it exists and is > 1 KB
    2. Attempt torch.load() to detect corruption
    3. Attempt model instantiation + load_state_dict
    4. Return structured result dict

    Returns:
        {
          "weights_exists":   bool,
          "weights_path":     str,
          "weights_size_kb":  float,
          "model_loaded":     bool,
          "ml_model_available": bool,  # True only if load fully succeeds
          "engine":           "ml_model" | "geometric_fallback",
          "model_type":       str,
          "device":           str,
          "param_count_m":    float,
          "is_stub":          bool,
          "fallback_reason":  str,
          "log":              str,
        }
    """
    result = {
        "weights_exists":     False,
        "weights_path":       str(_WEIGHTS_PATH.resolve()),
        "weights_size_kb":    0.0,
        "model_loaded":       False,
        "ml_model_available": False,
        "engine":             "geometric_fallback",
        "model_type":         "SareeTryOnModel",
        "device":             "cpu",
        "param_count_m":      0.0,
        "is_stub":            False,
        "fallback_reason":    "",
        "log":                "",
    }

    # ── 1. File check ────────────────────────────────────────────────────────
    if not _WEIGHTS_PATH.exists():
        msg = f"Weights file not found at {_WEIGHTS_PATH}"
        print(f"[ML Verify] {msg}")
        result["fallback_reason"] = msg
        result["log"] = msg
        return result

    size_kb = _WEIGHTS_PATH.stat().st_size / 1024
    result["weights_exists"]  = True
    result["weights_size_kb"] = round(size_kb, 1)

    if size_kb < 1:
        msg = f"Weights file too small ({size_kb:.1f} KB) — likely empty/corrupt"
        print(f"[ML Verify] {msg}")
        result["fallback_reason"] = msg
        result["log"] = msg
        return result

    # ── 2. torch.load check ──────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        msg = "PyTorch not installed — using geometric fallback"
        print(f"[ML Verify] {msg}")
        result["fallback_reason"] = msg
        result["log"] = msg
        return result

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result["device"] = str(device)
        ckpt = torch.load(str(_WEIGHTS_PATH), map_location=device, weights_only=False)
    except Exception as exc:
        msg = f"torch.load failed — corrupted checkpoint: {exc}"
        print(f"[ML Verify] {msg}")
        result["fallback_reason"] = msg
        result["log"] = msg
        return result

    # Detect stub checkpoints saved by train_routes.py
    if ckpt.get("stub"):
        msg = "Stub weights detected (no real training data) — using geometric fallback"
        print(f"[ML Verify] {msg}")
        result["is_stub"]         = True
        result["fallback_reason"] = msg
        result["log"]             = msg
        return result

    # ── 3. Model instantiation + state_dict load ─────────────────────────────
    try:
        from ai_engine.ml_models import SareeTryOnModel

        if _CONFIG_PATH.exists():
            import json as _json
            cfg = _json.loads(_CONFIG_PATH.read_text())
        else:
            cfg = _DEFAULT_CONFIG.copy()

        model = SareeTryOnModel(
            img_h=cfg.get("img_h", 256),
            img_w=cfg.get("img_w", 192),
            grid_size=cfg.get("grid_size", 5),
            base_ch=cfg.get("base_ch", 64),
        ).to(device)

        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()

        params = sum(p.numel() for p in model.parameters()) / 1e6
        result["param_count_m"] = round(params, 2)

        msg = (f"ML weights loaded successfully — {params:.1f}M params on {device}")
        print(f"[ML Verify] {msg}")
        result.update(
            model_loaded=True,
            ml_model_available=True,
            engine="ml_model",
            log=msg,
        )
        return result

    except Exception as exc:
        msg = f"Model state_dict load failed: {exc}"
        print(f"[ML Verify] {msg}")
        result["fallback_reason"] = msg
        result["log"] = msg
        return result


def run_inference_test() -> dict:
    """
    Lightweight synthetic inference test:
    - Loads model (or reuses cache)
    - Runs a single forward pass with random tensors
    - Returns pass/fail + timing

    Returns:
        { "passed": bool, "latency_ms": float, "error": str, "engine": str }
    """
    import time

    status = verify_weights()
    if not status["ml_model_available"]:
        return {
            "passed":     False,
            "latency_ms": 0.0,
            "error":      status["fallback_reason"] or "ML weights unavailable",
            "engine":     "geometric_fallback",
        }

    try:
        import torch
        from ai_engine.ml_models import SareeTryOnModel

        if _CONFIG_PATH.exists():
            import json as _json
            cfg = _json.loads(_CONFIG_PATH.read_text())
        else:
            cfg = _DEFAULT_CONFIG.copy()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = SareeTryOnModel(
            img_h=cfg.get("img_h", 256),
            img_w=cfg.get("img_w", 192),
            grid_size=cfg.get("grid_size", 5),
            base_ch=cfg.get("base_ch", 64),
        ).to(device)

        ckpt  = torch.load(str(_WEIGHTS_PATH), map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()

        h, w = cfg.get("img_h", 256), cfg.get("img_w", 192)
        with torch.no_grad():
            t0  = time.perf_counter()
            out = model(
                person       = torch.randn(1, 3, h, w, device=device),
                saree        = torch.randn(1, 3, h, w, device=device),
                blouse       = torch.randn(1, 3, h, w, device=device),
                pose_heatmap = torch.randn(1, 18, h, w, device=device),
                seg_mask     = torch.randn(1, 1, h, w, device=device),
                fabric_idx   = torch.tensor([0], device=device),
                style_idx    = torch.tensor([0], device=device),
            )
            latency_ms = (time.perf_counter() - t0) * 1000

        if "rendered" not in out:
            raise ValueError(f"Unexpected model output keys: {list(out.keys())}")

        msg = f"Inference test passed — {latency_ms:.1f} ms on {device}"
        print(f"[ML Verify] {msg}")
        return {
            "passed":     True,
            "latency_ms": round(latency_ms, 1),
            "error":      "",
            "engine":     "ml_model",
        }

    except Exception as exc:
        msg = f"Inference test FAILED: {exc} — switching to geometric fallback"
        print(f"[ML Verify] {msg}")
        # Invalidate singleton so next request uses fallback
        global _model
        _model = None
        return {
            "passed":     False,
            "latency_ms": 0.0,
            "error":      str(exc),
            "engine":     "geometric_fallback",
        }


def load_trained_model() -> bool:
    """
    Load SareeTryOnModel from weights file into GPU/CPU singleton.
    Returns True on success, False if weights not found or load fails.
    """
    global _model, _device, _config

    if _model is not None:
        return True

    if not weights_available():
        print(f"[ML Inference] No weights at {_WEIGHTS_PATH}. Using geometric fallback.")
        return False

    try:
        import torch
        from ai_engine.ml_models import SareeTryOnModel

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ML Inference] Loading model on {_device}…")

        # Load config
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                _config = json.load(f)
        else:
            _config = _DEFAULT_CONFIG.copy()

        cfg = _config
        _model = SareeTryOnModel(
            img_h=cfg["img_h"], img_w=cfg["img_w"],
            grid_size=cfg["grid_size"], base_ch=cfg["base_ch"],
        ).to(_device)

        ckpt = torch.load(str(_WEIGHTS_PATH), map_location=_device, weights_only=True)
        state = ckpt.get("model_state", ckpt)   # support both raw and wrapped checkpoints
        _model.load_state_dict(state, strict=False)
        _model.eval()

        params = sum(p.numel() for p in _model.parameters()) / 1e6
        print(f"[ML Inference] Model loaded ({params:.1f}M params) on {_device}")
        return True

    except Exception as exc:
        print(f"[ML Inference] Load failed: {exc}")
        _model = None
        return False


def generate_pose_heatmap(
    keypoints: dict,
    img_h: int,
    img_w: int,
    sigma: float = 8.0,
) -> np.ndarray:
    """
    Convert MediaPipe keypoint dict → 18-channel float32 Gaussian heatmap.

    Args:
        keypoints: {name: {x, y, visibility}} in pixel coords of original image.
        img_h, img_w: target heatmap spatial dimensions.
        sigma: Gaussian spread in pixels.

    Returns:
        np.ndarray shape (18, img_h, img_w), float32, values in [0, 1].
    """
    heatmap = np.zeros((18, img_h, img_w), dtype=np.float32)
    xs, ys  = np.mgrid[0:img_w, 0:img_h].astype(np.float32)  # W x H each

    for i, kp_name in enumerate(_HEATMAP_KPS):
        kp = keypoints.get(kp_name)
        if kp is None:
            continue
        vis = float(kp.get("visibility", 1.0))
        if vis < 0.15:
            continue
        cx, cy = float(kp["x"]), float(kp["y"])
        gauss  = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        heatmap[i] = gauss.T          # transpose to H x W

    return heatmap


def generate_segmentation_mask(
    body_image_path: str,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Generate a (1, img_h, img_w) float32 body segmentation mask in [0, 1].
    Uses rembg if available, otherwise GrabCut.
    """
    from ai_engine.segmentation import get_body_mask
    raw_mask = get_body_mask(body_image_path)               # H x W uint8 0/255
    resized  = cv2.resize(raw_mask, (img_w, img_h),
                          interpolation=cv2.INTER_NEAREST)
    return (resized.astype(np.float32) / 255.0)[np.newaxis]  # 1 x H x W


def preprocess_inputs(
    body_image_path: str,
    saree_image_path: str,
    blouse_image_path: str,
    pose_data: dict,
) -> dict:
    """
    Load images + pose data and convert to normalised float32 NumPy arrays
    ready to be stacked into PyTorch tensors.

    Returns dict with keys:
        person_np      (3, H, W)  float32 -1..1
        saree_np       (3, H, W)  float32 -1..1
        blouse_np      (3, H, W)  float32 -1..1
        pose_np        (18, H, W) float32  0..1
        seg_np         (1, H, W)  float32  0..1
    """
    cfg  = _config or _DEFAULT_CONFIG
    h, w = cfg.get("img_h", 256), cfg.get("img_w", 192)
    mean = np.array(cfg.get("normalize_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std  = np.array(cfg.get("normalize_std",  [0.229, 0.224, 0.225]), dtype=np.float32)

    def _load_rgb(path: str) -> np.ndarray:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {path}")
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
        arr     = resized.astype(np.float32) / 255.0       # H x W x 3
        arr     = (arr - mean) / std                       # normalise
        return arr.transpose(2, 0, 1)                      # 3 x H x W

    # Scale pose keypoints to model input resolution
    orig_size = pose_data.get("image_size", {})
    orig_w    = orig_size.get("width",  w)
    orig_h    = orig_size.get("height", h)
    keypoints = {}
    for name, kp in pose_data.get("keypoints", {}).items():
        keypoints[name] = {
            "x":          kp["x"] * w / orig_w,
            "y":          kp["y"] * h / orig_h,
            "visibility": kp.get("visibility", 1.0),
        }

    return {
        "person_np": _load_rgb(body_image_path),
        "saree_np":  _load_rgb(saree_image_path),
        "blouse_np": _load_rgb(blouse_image_path),
        "pose_np":   generate_pose_heatmap(keypoints, h, w),
        "seg_np":    generate_segmentation_mask(body_image_path, h, w),
    }


def run_ml_tryon(
    body_image_path:   str,
    saree_image_path:  str,
    blouse_image_path: str,
    pose_data:         dict,
    saree_features:    dict,
    blouse_config:     dict,
    draping_style:     str,
    output_folder:     str,
) -> str:
    """
    Run ML virtual try-on pipeline.

    Returns:
        Absolute path to saved output JPEG.

    Raises:
        RuntimeError if model not loaded.
    """
    if not load_trained_model():
        raise RuntimeError("ML weights unavailable — use geometric fallback.")

    import torch
    from ai_engine.ml_models import SareeTryOnModel

    cfg  = _config or _DEFAULT_CONFIG
    h, w = cfg.get("img_h", 256), cfg.get("img_w", 192)

    # ── Preprocess ────────────────────────────────────────────────────────────
    inputs = preprocess_inputs(
        body_image_path, saree_image_path, blouse_image_path, pose_data
    )

    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0).to(_device)  # 1 x C x H x W

    person_t = _to_tensor(inputs["person_np"])
    saree_t  = _to_tensor(inputs["saree_np"])
    blouse_t = _to_tensor(inputs["blouse_np"])
    pose_t   = _to_tensor(inputs["pose_np"])
    seg_t    = _to_tensor(inputs["seg_np"])

    fabric_name  = saree_features.get("fabric_type", "silk")
    fabric_t     = torch.tensor([SareeTryOnModel.fabric_index(fabric_name)],
                                 dtype=torch.long, device=_device)
    style_t      = torch.tensor([SareeTryOnModel.style_index(draping_style)],
                                 dtype=torch.long, device=_device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        out = _model(
            person=person_t, saree=saree_t, blouse=blouse_t,
            pose_heatmap=pose_t, seg_mask=seg_t,
            fabric_idx=fabric_t, style_idx=style_t,
        )

    # ── Postprocess ───────────────────────────────────────────────────────────
    result_bgr = postprocess_output(out["rendered"], body_image_path)

    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"output_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(out_path, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out_path


def postprocess_output(
    rendered_tensor,          # (1, 3, H, W) torch tensor in [-1, 1]
    body_image_path: str,     # used to match output size to input
) -> np.ndarray:
    """
    Convert model output tensor → BGR uint8 numpy array at original body size.
    Applies mild sharpening and saturation boost.
    """
    arr = rendered_tensor[0].cpu().permute(1, 2, 0).numpy()  # H x W x 3
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Resize back to original body image dimensions
    body_bgr = cv2.imread(body_image_path)
    if body_bgr is not None:
        oh, ow = body_bgr.shape[:2]
        bgr = cv2.resize(bgr, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

    # Mild sharpening
    blurred  = cv2.GaussianBlur(bgr, (0, 0), 2)
    bgr      = cv2.addWeighted(bgr, 1.3, blurred, -0.3, 0)

    # Slight saturation boost
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr
