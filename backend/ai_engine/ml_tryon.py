"""
ML Try-On Inference Module
==========================
Loads trained GMM + TOM weights (from Kaggle training) and runs
the full virtual try-on pipeline.

Falls back gracefully if weights are not present.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path

# Weight paths — place exported weights from Kaggle here
_MODELS_DIR = Path(__file__).parent.parent / "models"
_GMM_PATH   = _MODELS_DIR / "gmm.pth"
_TOM_PATH   = _MODELS_DIR / "tom.pth"
_CFG_PATH   = _MODELS_DIR / "config.json"

_gmm = None
_tom = None
_cfg = None
_device = None


def weights_available() -> bool:
    return _GMM_PATH.exists() and _TOM_PATH.exists()


def _load_models():
    global _gmm, _tom, _cfg, _device
    if _gmm is not None:
        return True

    if not weights_available():
        return False

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ML TryOn] Loading models on {_device}")

        # Load config
        if _CFG_PATH.exists():
            with open(_CFG_PATH) as f:
                _cfg = json.load(f)
        else:
            _cfg = {"gmm": {"grid_size": 5, "img_h": 256, "img_w": 192},
                    "tom": {"in_ch": 24, "base_ch": 64},
                    "img_h": 256, "img_w": 192,
                    "normalize_mean": [0.5, 0.5, 0.5],
                    "normalize_std":  [0.5, 0.5, 0.5]}

        # Import model classes from training notebook equivalents
        from ai_engine.ml_models import GMM, TryOnModule

        gmm_cfg = _cfg["gmm"]
        _gmm = GMM(grid_size=gmm_cfg["grid_size"],
                   img_h=gmm_cfg["img_h"],
                   img_w=gmm_cfg["img_w"]).to(_device)
        _gmm.load_state_dict(
            torch.load(str(_GMM_PATH), map_location=_device, weights_only=True)
        )
        _gmm.eval()

        tom_cfg = _cfg["tom"]
        _tom = TryOnModule(in_ch=tom_cfg["in_ch"],
                           base_ch=tom_cfg["base_ch"]).to(_device)
        _tom.load_state_dict(
            torch.load(str(_TOM_PATH), map_location=_device, weights_only=True)
        )
        _tom.eval()

        print("[ML TryOn] Models loaded successfully")
        return True

    except Exception as e:
        print(f"[ML TryOn] Failed to load models: {e}")
        _gmm = _tom = None
        return False


def _make_pose_heatmap(keypoints: dict, h: int, w: int, sigma: float = 8.0) -> np.ndarray:
    """Generate 18-channel Gaussian heatmap from pose keypoints."""
    DRAPING_KPS = [
        "left_shoulder", "right_shoulder", "left_elbow",  "right_elbow",
        "left_wrist",    "right_wrist",    "left_hip",    "right_hip",
        "left_knee",     "right_knee",     "left_ankle",  "right_ankle",
        "nose",          "left_ear",       "right_ear",
        "left_eye",      "right_eye",      "mouth_left",
    ]
    heatmaps = np.zeros((18, h, w), dtype=np.float32)
    ys, xs   = np.mgrid[0:h, 0:w].astype(np.float32)

    for i, kp_name in enumerate(DRAPING_KPS):
        if kp_name not in keypoints:
            continue
        kp  = keypoints[kp_name]
        vis = kp.get("visibility", 1.0)
        if vis < 0.2:
            continue
        cx, cy = float(kp["x"]), float(kp["y"])
        heatmaps[i] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))

    return heatmaps


def run_ml_tryon(
    body_image_path: str,
    saree_image_path: str,
    pose_data: dict,
    output_folder: str,
) -> str:
    """
    Run the ML virtual try-on pipeline.

    Args:
        body_image_path:  Path to body image.
        saree_image_path: Path to saree/garment image.
        pose_data:        Output from pose_detection.detect_pose().
        output_folder:    Directory to save output.

    Returns:
        Path to composed output image.

    Raises:
        RuntimeError if ML weights are unavailable.
    """
    if not _load_models():
        raise RuntimeError("ML model weights not found. Run Kaggle training first.")

    import torch
    import torch.nn.functional as F
    import uuid

    cfg    = _cfg
    img_h  = cfg["img_h"]
    img_w  = cfg["img_w"]
    mean   = cfg["normalize_mean"]
    std    = cfg["normalize_std"]

    # ── Load + preprocess images ──────────────────────────────────────────────
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    person_pil  = Image.open(body_image_path).convert("RGB")
    garment_pil = Image.open(saree_image_path).convert("RGB")
    orig_w, orig_h = person_pil.size

    person_t  = transform(person_pil).unsqueeze(0).to(_device)   # 1 x 3 x H x W
    garment_t = transform(garment_pil).unsqueeze(0).to(_device)  # 1 x 3 x H x W

    # ── Build pose heatmap ────────────────────────────────────────────────────
    keypoints = pose_data.get("keypoints", {})
    orig_size = pose_data.get("image_size", {"width": orig_w, "height": orig_h})

    # Scale keypoints to model input size
    sx = img_w / orig_size.get("width",  orig_w)
    sy = img_h / orig_size.get("height", orig_h)
    scaled_kp = {}
    for name, kp in keypoints.items():
        scaled_kp[name] = {
            "x":          kp["x"] * sx,
            "y":          kp["y"] * sy,
            "visibility": kp.get("visibility", 1.0),
        }

    heatmap = _make_pose_heatmap(scaled_kp, img_h, img_w)   # 18 x H x W
    heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).to(_device)  # 1 x 18 x H x W

    # ── GMM: warp garment ─────────────────────────────────────────────────────
    person_pose = torch.cat([person_t, heatmap_t], dim=1)   # 1 x 21 x H x W
    with torch.no_grad():
        warped_garment, _ = _gmm(person_pose, garment_t)    # 1 x 3 x H x W

    # ── TOM: composite ────────────────────────────────────────────────────────
    tom_input = torch.cat([person_t, warped_garment, heatmap_t], dim=1)  # 1 x 24
    with torch.no_grad():
        composed, alpha = _tom(tom_input)                   # 1 x 3, 1 x 1

    # ── Convert back to BGR image ─────────────────────────────────────────────
    def tensor_to_bgr(t: "torch.Tensor") -> np.ndarray:
        arr = t[0].cpu().permute(1, 2, 0).numpy()          # H x W x 3
        arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    composed_bgr = tensor_to_bgr(composed)

    # Resize back to original body image size
    body_bgr = cv2.imread(body_image_path)
    out_h, out_w = body_bgr.shape[:2]
    composed_bgr = cv2.resize(composed_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"output_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(out_path, composed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return out_path
