"""
Pose detection routes: run MediaPipe pose estimation on a body image.
"""

import os
from flask import Blueprint, request, current_app
from utils.helpers import success_response, error_response, token_required
pose_bp = Blueprint("pose", __name__)


@pose_bp.route("/detect", methods=["POST"])
@token_required
def detect():
    """
    Run pose detection on the provided body image.
    Body: { "body_image_path": "<absolute path on server>" }
    """
    data = request.get_json()
    if not data or "body_image_path" not in data:
        return error_response("body_image_path is required")

    image_path = os.path.normpath(data["body_image_path"])
    try:
        from ai_engine.pose_detection import detect_pose
        pose_data = detect_pose(image_path)
        if pose_data is None:
            return error_response("Could not detect pose. Ensure the full body is visible.", 422)
        return success_response(pose_data, "Pose detected successfully")
    except FileNotFoundError:
        return error_response("Image file not found", 404)
    except Exception as e:
        return error_response(f"Pose detection failed: {str(e)}", 500)
