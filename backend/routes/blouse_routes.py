"""
Blouse customization routes.
"""

from flask import Blueprint, request, current_app
from bson import ObjectId
from utils.helpers import (
    success_response, error_response, token_required, admin_required,
)
from models.schemas import blouse_schema

blouse_bp = Blueprint("blouse", __name__)

VALID_NECK_TYPES   = ["round", "boat", "v_neck", "square", "sweetheart"]
VALID_SLEEVE_TYPES = ["sleeveless", "short", "half", "full"]


@blouse_bp.route("/customize", methods=["POST"])
@token_required
def customize_blouse():
    """
    Fit blouse onto the body using pose data and customization options.
    Body: {
        "body_image_path": "...",
        "blouse_image_path": "...",
        "pose_data": {...},
        "neck_type": "round",
        "sleeve_type": "short",
        "color": "#FF0000"
    }
    """
    data = request.get_json()
    required = ["body_image_path", "blouse_image_path", "pose_data"]
    for field in required:
        if field not in data:
            return error_response(f"{field} is required")

    neck_type   = data.get("neck_type", "round")
    sleeve_type = data.get("sleeve_type", "short")
    color       = data.get("color", "#FFFFFF")

    if neck_type not in VALID_NECK_TYPES:
        return error_response(f"Invalid neck_type. Valid: {VALID_NECK_TYPES}")
    if sleeve_type not in VALID_SLEEVE_TYPES:
        return error_response(f"Invalid sleeve_type. Valid: {VALID_SLEEVE_TYPES}")

    try:
        from ai_engine.blouse_fitting import fit_blouse
        result = fit_blouse(
            body_image_path=data["body_image_path"],
            blouse_image_path=data["blouse_image_path"],
            pose_data=data["pose_data"],
            neck_type=neck_type,
            sleeve_type=sleeve_type,
            color=color,
        )
        return success_response(result, "Blouse fitted successfully")
    except Exception as e:
        return error_response(f"Blouse fitting failed: {str(e)}", 500)


@blouse_bp.route("/templates", methods=["GET"])
def list_blouse_templates():
    """Return all blouse templates."""
    db   = current_app.mongo.db
    docs = list(db.blouses.find({"is_template": True}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return success_response(docs)


@blouse_bp.route("/templates", methods=["POST"])
@admin_required
def add_blouse_template():
    """Admin: add a blouse template."""
    data = request.get_json()
    if not data:
        return error_response("No data provided")

    doc = blouse_schema(
        name=data.get("name", "Unnamed Blouse"),
        image_path=data.get("image_path", ""),
        neck_type=data.get("neck_type", "round"),
        sleeve_type=data.get("sleeve_type", "short"),
        uploaded_by=request.user_id,
    )
    result = current_app.mongo.db.blouses.insert_one(doc)
    return success_response({"id": str(result.inserted_id)}, "Blouse template added", 201)
