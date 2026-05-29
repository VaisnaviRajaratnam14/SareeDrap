"""
Saree processing routes: extract texture/border, list templates.
"""

from flask import Blueprint, request, current_app
from bson import ObjectId
from utils.helpers import (
    success_response, error_response, token_required, admin_required,
)
from models.schemas import saree_schema

saree_bp = Blueprint("saree", __name__)


@saree_bp.route("/process", methods=["POST"])
@token_required
def process_saree():
    """
    Extract fabric texture, border, and dominant colour from a saree image.
    Body: { "saree_image_path": "...", "fabric_type": "silk" }
    """
    data = request.get_json()
    if not data or "saree_image_path" not in data:
        return error_response("saree_image_path is required")

    image_path  = data["saree_image_path"]
    fabric_type = data.get("fabric_type", "silk")

    try:
        from ai_engine.saree_draping import extract_saree_features
        features = extract_saree_features(image_path, fabric_type)
        return success_response(features, "Saree features extracted")
    except FileNotFoundError:
        return error_response("Saree image not found", 404)
    except Exception as e:
        return error_response(f"Saree processing failed: {str(e)}", 500)


@saree_bp.route("/templates", methods=["GET"])
def list_templates():
    """Return all saree templates available in the system."""
    db   = current_app.mongo.db
    docs = list(db.sarees.find({"is_template": True}))
    for d in docs:
        d["_id"] = str(d["_id"])
    return success_response(docs)


@saree_bp.route("/templates", methods=["POST"])
@admin_required
def add_template():
    """Admin: add a new saree template record."""
    data = request.get_json()
    if not data:
        return error_response("No data provided")

    doc = saree_schema(
        name=data.get("name", "Unnamed Saree"),
        image_path=data.get("image_path", ""),
        fabric_type=data.get("fabric_type", "silk"),
        draping_style=data.get("draping_style", "nivi"),
        border_color=data.get("border_color", ""),
        uploaded_by=request.user_id,
    )
    result = current_app.mongo.db.sarees.insert_one(doc)
    return success_response({"id": str(result.inserted_id)}, "Template added", 201)


@saree_bp.route("/templates/<template_id>", methods=["DELETE"])
@admin_required
def delete_template(template_id):
    """Admin: delete a saree template."""
    db = current_app.mongo.db
    result = db.sarees.delete_one({"_id": ObjectId(template_id)})
    if result.deleted_count == 0:
        return error_response("Template not found", 404)
    return success_response(None, "Template deleted")
