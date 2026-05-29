"""
/api/process  — unified draping pipeline endpoint.

Checks for trained ML weights first.
If weights exist   → uses SareeTryOnModel (VITON-HD style).
If weights absent  → falls back to the geometric OpenCV engine.

Response always includes "engine_used": "ml_model" | "geometric_fallback"
"""

import os
from flask import Blueprint, request, current_app
from bson import ObjectId
from utils.helpers import success_response, error_response, token_required
from models.schemas import output_schema

process_bp = Blueprint("process", __name__)


@process_bp.route("/run", methods=["POST"])
@token_required
def run_pipeline():
    """
    Full draping pipeline — ML or geometric fallback.

    Request body (JSON):
    {
        "body_image_path":   "<abs path>",
        "saree_image_path":  "<abs path>",
        "blouse_image_path": "<abs path>",
        "pose_data":         { ... },          // from /api/pose/detect
        "saree_features":    { ... },          // from /api/saree/process
        "blouse_config":     { "neck_type", "sleeve_type", "color" },
        "draping_style":     "nivi" | "bridal" | "hanging" | "gujarati"
    }
    """
    data = request.get_json()

    required = [
        "body_image_path", "saree_image_path",
        "blouse_image_path", "pose_data", "draping_style",
    ]
    for field in required:
        if not data or field not in data:
            return error_response(f"{field} is required")

    body_path   = os.path.normpath(data["body_image_path"])
    saree_path  = os.path.normpath(data["saree_image_path"])
    blouse_path = os.path.normpath(data["blouse_image_path"])

    for label, path in [("body",   body_path),
                         ("saree",  saree_path),
                         ("blouse", blouse_path)]:
        if not os.path.exists(path):
            return error_response(f"{label} image not found: {path}", 404)

    pose_data     = data["pose_data"]
    saree_features = data.get("saree_features", {})
    blouse_config  = data.get("blouse_config",  {})
    draping_style  = data["draping_style"]
    output_folder  = current_app.config["OUTPUT_FOLDER"]

    engine_used = "geometric_fallback"
    output_path = None

    # ── Try ML model first ─────────────────────────────────────────────────────
    try:
        from ai_engine.ml_saree_inference import weights_available, run_ml_tryon
        if weights_available():
            output_path = run_ml_tryon(
                body_image_path=body_path,
                saree_image_path=saree_path,
                blouse_image_path=blouse_path,
                pose_data=pose_data,
                saree_features=saree_features,
                blouse_config=blouse_config,
                draping_style=draping_style,
                output_folder=output_folder,
            )
            engine_used = "ml_model"
    except Exception as ml_err:
        print(f"[Process] ML engine failed: {ml_err} — falling back to geometric engine")
        output_path = None

    # ── Geometric fallback ─────────────────────────────────────────────────────
    if output_path is None:
        try:
            from ai_engine.rendering import render_draped_image
            output_path = render_draped_image(
                body_image_path=body_path,
                saree_image_path=saree_path,
                blouse_image_path=blouse_path,
                pose_data=pose_data,
                saree_features=saree_features,
                blouse_config=blouse_config,
                draping_style=draping_style,
                output_folder=output_folder,
            )
            engine_used = "geometric_fallback"
        except Exception as geo_err:
            return error_response(f"Draping failed: {str(geo_err)}", 500)

    # ── Persist to MongoDB (optional — skip if DB unavailable) ────────────────
    output_id = None
    try:
        if current_app.db_available:
            db  = current_app.mongo.db
            doc = output_schema(
                user_id=request.user_id,
                body_image=body_path,
                saree_image=saree_path,
                blouse_image=blouse_path,
                output_image=output_path,
                draping_style=draping_style,
                blouse_config=blouse_config,
            )
            doc["pose_data"]    = pose_data
            doc["engine_used"]  = engine_used
            result   = db.generated_outputs.insert_one(doc)
            output_id = str(result.inserted_id)
    except Exception as db_err:
        print(f"[Process] DB save failed (non-fatal): {db_err}")

    return success_response(
        {
            "output_id":    output_id,
            "output_path":  output_path,
            "filename":     os.path.basename(output_path),
            "engine_used":  engine_used,
        },
        "Draping complete",
        201,
    )


@process_bp.route("/engine-status", methods=["GET"])
def engine_status():
    """Returns which engine will be used for the next request."""
    try:
        from ai_engine.ml_saree_inference import weights_available
        ml_ready = weights_available()
    except Exception:
        ml_ready = False

    return success_response({
        "ml_model_ready":     ml_ready,
        "engine":             "ml_model" if ml_ready else "geometric_fallback",
        "weights_path":       str(
            ((__import__("pathlib").Path(__file__).parent.parent) /
             "dataset" / "trained_models" / "saree_tryon_model.pth").resolve()
        ),
    })
