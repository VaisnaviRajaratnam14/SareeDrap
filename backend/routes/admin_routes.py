"""
Admin dashboard routes: manage users, sarees, blouses, outputs, datasets.
"""

from flask import Blueprint, request, current_app
from bson import ObjectId
from utils.helpers import success_response, error_response, admin_required

admin_bp = Blueprint("admin", __name__)


# ── Stats ──────────────────────────────────────────────────────────────────────
@admin_bp.route("/stats", methods=["GET"])
@admin_required
def stats():
    """Return summary counts for the admin dashboard."""
    db = current_app.mongo.db
    return success_response({
        "total_users":   db.users.count_documents({}),
        "total_sarees":  db.sarees.count_documents({}),
        "total_blouses": db.blouses.count_documents({}),
        "total_outputs": db.generated_outputs.count_documents({"is_deleted": False}),
    })


# ── Users ──────────────────────────────────────────────────────────────────────
@admin_bp.route("/users", methods=["GET"])
@admin_required
def list_users():
    """List all users (passwords excluded)."""
    db   = current_app.mongo.db
    users = list(db.users.find({}, {"password_hash": 0}))
    for u in users:
        u["_id"] = str(u["_id"])
        u["created_at"] = str(u.get("created_at"))
    return success_response(users)


@admin_bp.route("/users/<user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    """Deactivate a user account."""
    db = current_app.mongo.db
    result = db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"is_active": False}},
    )
    if result.matched_count == 0:
        return error_response("User not found", 404)
    return success_response(None, "User deactivated")


# ── Outputs ────────────────────────────────────────────────────────────────────
@admin_bp.route("/outputs", methods=["GET"])
@admin_required
def list_outputs():
    """List all generated outputs."""
    db   = current_app.mongo.db
    docs = list(
        db.generated_outputs.find({"is_deleted": False}, {"pose_data": 0})
        .sort("created_at", -1)
        .limit(50)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
        d["created_at"] = str(d.get("created_at"))
    return success_response(docs)


@admin_bp.route("/outputs/<output_id>", methods=["DELETE"])
@admin_required
def delete_output(output_id):
    """Admin hard-delete an output record."""
    db     = current_app.mongo.db
    result = db.generated_outputs.delete_one({"_id": ObjectId(output_id)})
    if result.deleted_count == 0:
        return error_response("Output not found", 404)
    return success_response(None, "Output deleted")


# ── Dataset management ─────────────────────────────────────────────────────────
@admin_bp.route("/dataset/info", methods=["GET"])
@admin_required
def dataset_info():
    """Return dataset directory listing info."""
    import os
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    raw_dir = os.path.join("dataset", "raw")
    info = {}
    for cat in ["body_images", "saree_images", "blouse_materials"]:
        path = os.path.join(raw_dir, cat)
        if os.path.isdir(path):
            info[cat] = sum(
                1 for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in exts
            )
        else:
            info[cat] = 0
    return success_response(info, "Dataset info retrieved")
