"""
Pose Detection Module
=====================
Uses MediaPipe Pose Landmarker (Tasks API, mediapipe >= 0.10) to detect
33 body landmarks from a body image.

Falls back to a proportional geometric estimation if the model file is
not yet downloaded — so the upload→customize flow always succeeds.
"""

import os
import urllib.request
import cv2
import numpy as np
from typing import Optional

# ── MediaPipe model path ───────────────────────────────────────────────────────
_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# Landmark name list matching the 33 MediaPipe pose landmarks (in order)
_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _ensure_model() -> bool:
    """Download the pose landmarker model if missing. Returns True on success."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if os.path.exists(_MODEL_PATH) and os.path.getsize(_MODEL_PATH) > 100_000:
        return True
    try:
        print(f"Downloading pose landmarker model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Pose landmarker model downloaded.")
        return True
    except Exception as e:
        print(f"Warning: could not download pose model: {e}")
        return False


def _detect_with_mediapipe(image_path: str, bgr: np.ndarray) -> Optional[dict]:
    """Try MediaPipe Tasks API pose detection."""
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        import mediapipe as mp

        if not _ensure_model():
            return None

        base_opts = mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH)
        options   = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_segmentation_masks=False,
        )

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        h, w = bgr.shape[:2]
        landmarks = result.pose_landmarks[0]
        keypoints = {}
        for idx, lm in enumerate(landmarks):
            name = _LANDMARK_NAMES[idx] if idx < len(_LANDMARK_NAMES) else f"lm_{idx}"
            keypoints[name] = {
                "x":          round(lm.x * w, 2),
                "y":          round(lm.y * h, 2),
                "z":          round(lm.z, 4),
                "visibility": round(getattr(lm, "visibility", 1.0), 4),
                "nx":         round(lm.x, 4),
                "ny":         round(lm.y, 4),
            }
        return keypoints

    except Exception as e:
        print(f"MediaPipe Tasks API failed: {e}")
        return None


def _geometric_fallback(bgr: np.ndarray) -> dict:
    """
    Proportional body landmark estimation using image dimensions.
    Used when MediaPipe model is unavailable.
    Assumes a standard standing full-body pose.
    """
    h, w = bgr.shape[:2]

    # Standard body proportion ratios (from top of head)
    cx = w * 0.50   # horizontal centre

    # Try to detect face/skin region to refine shoulder position
    face_top = h * 0.03
    shoulder_y = h * 0.22

    pts = {
        "nose":            (cx,        h * 0.05),
        "left_shoulder":   (cx - w*0.13, shoulder_y),
        "right_shoulder":  (cx + w*0.13, shoulder_y),
        "left_elbow":      (cx - w*0.18, h * 0.38),
        "right_elbow":     (cx + w*0.18, h * 0.38),
        "left_wrist":      (cx - w*0.20, h * 0.52),
        "right_wrist":     (cx + w*0.20, h * 0.52),
        "left_hip":        (cx - w*0.10, h * 0.52),
        "right_hip":       (cx + w*0.10, h * 0.52),
        "left_knee":       (cx - w*0.09, h * 0.70),
        "right_knee":      (cx + w*0.09, h * 0.70),
        "left_ankle":      (cx - w*0.08, h * 0.90),
        "right_ankle":     (cx + w*0.08, h * 0.90),
        "left_ear":        (cx - w*0.05, h * 0.07),
        "right_ear":       (cx + w*0.05, h * 0.07),
        "left_eye":        (cx - w*0.03, h * 0.04),
        "right_eye":       (cx + w*0.03, h * 0.04),
        "mouth_left":      (cx - w*0.02, h * 0.09),
        "mouth_right":     (cx + w*0.02, h * 0.09),
        "left_heel":       (cx - w*0.08, h * 0.93),
        "right_heel":      (cx + w*0.08, h * 0.93),
        "left_foot_index": (cx - w*0.07, h * 0.97),
        "right_foot_index":(cx + w*0.07, h * 0.97),
    }

    keypoints = {}
    for name, (px, py) in pts.items():
        keypoints[name] = {
            "x": round(float(px), 2), "y": round(float(py), 2),
            "z": 0.0, "visibility": 0.85,
            "nx": round(float(px) / w, 4), "ny": round(float(py) / h, 4),
        }
    return keypoints


def detect_pose(image_path: str) -> Optional[dict]:
    """
    Run pose detection on an image file.

    Tries MediaPipe Tasks API first; falls back to geometric estimation.

    Args:
        image_path: Absolute path to the body image.

    Returns:
        dict with keypoints, image_size, draping_anchors, and detection_method.

    Raises:
        FileNotFoundError: if image_path does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = bgr.shape[:2]

    # Try MediaPipe Tasks API
    keypoints = _detect_with_mediapipe(image_path, bgr)
    method    = "mediapipe"

    # Fall back to geometric estimation
    if keypoints is None:
        print("Using geometric fallback for pose estimation.")
        keypoints = _geometric_fallback(bgr)
        method    = "geometric_fallback"

    anchors = _compute_draping_anchors(keypoints, w, h)

    return {
        "keypoints":        keypoints,
        "image_size":       {"width": w, "height": h},
        "draping_anchors":  anchors,
        "detection_method": method,
    }


def _compute_draping_anchors(kp: dict, w: int, h: int) -> dict:
    """Compute higher-level anchor points for the draping engine (pixel coords)."""

    def midpoint(a: str, b: str) -> dict:
        ax, ay = kp[a]["x"], kp[a]["y"]
        bx, by = kp[b]["x"], kp[b]["y"]
        return {"x": round((ax + bx) / 2, 2), "y": round((ay + by) / 2, 2)}

    def pt(name: str) -> dict:
        return {"x": kp[name]["x"], "y": kp[name]["y"]}

    def safe_pt(name: str, default_x: float, default_y: float) -> dict:
        if name in kp:
            return pt(name)
        return {"x": default_x, "y": default_y}

    anchors = {
        "shoulder_center": midpoint("left_shoulder", "right_shoulder"),
        "waist_center":    midpoint("left_hip", "right_hip"),
        "left_shoulder":   pt("left_shoulder"),
        "right_shoulder":  pt("right_shoulder"),
        "left_hip":        pt("left_hip"),
        "right_hip":       pt("right_hip"),
        "left_wrist":      safe_pt("left_wrist",  w * 0.2, h * 0.52),
        "right_wrist":     safe_pt("right_wrist", w * 0.8, h * 0.52),
        "left_elbow":      safe_pt("left_elbow",  w * 0.18, h * 0.38),
        "right_elbow":     safe_pt("right_elbow", w * 0.82, h * 0.38),
        "left_ankle":      safe_pt("left_ankle",  w * 0.42, h * 0.90),
        "right_ankle":     safe_pt("right_ankle", w * 0.58, h * 0.90),
    }

    sh = anchors["shoulder_center"]
    wa = anchors["waist_center"]
    la = anchors["left_ankle"]

    anchors["shoulder_width"] = round(
        abs(kp["left_shoulder"]["x"] - kp["right_shoulder"]["x"]), 2
    )
    anchors["torso_height"] = round(abs(sh["y"] - wa["y"]), 2)
    anchors["body_height"]  = round(abs(sh["y"] - la["y"]), 2)

    return anchors


def draw_pose_landmarks(image_path: str, output_path: str) -> str:
    """Draw detected landmarks on image and save. For debugging."""
    bgr = cv2.imread(image_path)
    result = detect_pose(image_path)
    if result and result.get("keypoints"):
        for name, lm in result["keypoints"].items():
            cv2.circle(bgr, (int(lm["x"]), int(lm["y"])), 4, (0, 255, 0), -1)
    cv2.imwrite(output_path, bgr)
    return output_path
