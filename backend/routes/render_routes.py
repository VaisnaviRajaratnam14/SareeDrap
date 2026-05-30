"""
Render routes: orchestrate the full AI draping pipeline and output download.
"""

import os
from pathlib import Path
from flask import Blueprint, request, current_app, send_file

from bson import ObjectId
from utils.helpers import (
    success_response, error_response, token_required, delete_file,
)
from models.schemas import output_schema

render_bp = Blueprint("render", __name__)

# Weights path — same as ml_saree_inference.py uses
_WEIGHTS_PATH = (
    Path(__file__).parent.parent / "dataset" / "trained_models" / "saree_tryon_model.pth"
)


def _ml_available() -> bool:
    """Return True only if the weights file exists AND is a valid .pth file (>1 KB)."""
    try:
        if not _WEIGHTS_PATH.exists():
            return False
        if _WEIGHTS_PATH.stat().st_size < 1024:
            return False
        return True
    except Exception:
        return False


@render_bp.route("/generate", methods=["POST"])
@token_required
def generate():
    """
    Full pipeline: drape saree + fit blouse → render final image.

    Tries ML model first when weights exist; falls back to geometric engine.
    Always returns:
      engine_used       – "ml_model" | "geometric_fallback"
      ml_model_available – true | false
      fallback_reason   – "" | reason string
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

    for label, path in [("body", body_path), ("saree", saree_path), ("blouse", blouse_path)]:
        if not os.path.exists(path):
            return error_response(f"{label} image not found: {path}", 404)

    pose_data      = data["pose_data"]
    saree_features = data.get("saree_features", {})
    blouse_config  = data.get("blouse_config",  {})
    draping_style  = data["draping_style"]
    output_folder  = current_app.config["OUTPUT_FOLDER"]

    ml_available   = _ml_available()
    engine_used    = "geometric_fallback"
    fallback_reason = ""
    output_path    = None

    # ── Attempt ML model ───────────────────────────────────────────────────────
    if ml_available:
        try:
            from ai_engine.ml_saree_inference import run_ml_tryon
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
            engine_used     = "ml_model"
            fallback_reason = ""
        except Exception as ml_err:
            import traceback as _tb
            fallback_reason = f"ML inference failed: {ml_err}"
            print(f"[Render] {fallback_reason} — using geometric fallback")
            print(f"[Render] ML traceback: {_tb.format_exc()}")
            output_path = None
    else:
        fallback_reason = (
            "trained weights not found or invalid — "
            f"place saree_tryon_model.pth in {_WEIGHTS_PATH.parent}"
        )

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
            print(f"[Render] engine=geometric_fallback output={output_path}")
        except Exception as geo_err:
            import traceback as _tb2
            print(f"[Render] Geometric fallback CRASHED:\n{_tb2.format_exc()}")
            return error_response(f"Draping failed: {str(geo_err)}", 500)

    # ── Persist to MongoDB (skipped if DB unavailable) ─────────────────────────
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
            result    = db.generated_outputs.insert_one(doc)
            output_id = str(result.inserted_id)
    except Exception as db_err:
        print(f"[Render] DB save failed (non-fatal): {db_err}")

    return success_response(
        {
            "output_id":         output_id,
            "output_path":       output_path,
            "filename":          os.path.basename(output_path),
            "engine_used":       engine_used,
            "ml_model_available": ml_available,
            "fallback_reason":   fallback_reason,
        },
        "Draping rendered successfully",
        201,
    )


@render_bp.route("/engine-status", methods=["GET"])
def engine_status():
    """
    Deep-verify the ML weights and return full engine status.
    Always calls verify_weights() — never relies on file-exists alone.
    """
    try:
        from ai_engine.ml_saree_inference import verify_weights
        status = verify_weights()
    except Exception as exc:
        status = {
            "weights_exists":     _ml_available(),
            "weights_path":       str(_WEIGHTS_PATH.resolve()),
            "weights_size_kb":    0.0,
            "model_loaded":       False,
            "ml_model_available": False,
            "engine":             "geometric_fallback",
            "model_type":         "SareeTryOnModel",
            "device":             "cpu",
            "param_count_m":      0.0,
            "is_stub":            False,
            "fallback_reason":    str(exc),
            "log":                str(exc),
        }

    if status["ml_model_available"]:
        print(f"[Engine] {status['log']}")
    else:
        print(f"[Engine] Using geometric fallback — {status['fallback_reason']}")

    return success_response(status)


@render_bp.route("/test-inference", methods=["POST"])
def test_inference():
    """
    Run a lightweight synthetic forward-pass test.
    If it fails the engine falls back automatically.
    """
    try:
        from ai_engine.ml_saree_inference import run_inference_test
        result = run_inference_test()
    except Exception as exc:
        result = {
            "passed":     False,
            "latency_ms": 0.0,
            "error":      str(exc),
            "engine":     "geometric_fallback",
        }
    status_code = 200 if result["passed"] else 200  # always 200 — caller reads "passed"
    return success_response(result)


@render_bp.route("/download/<output_id>", methods=["GET"])
@token_required
def download(output_id):
    """Download the final rendered output image."""
    # ── No-DB / demo mode: serve from outputs folder by filename ───────────────
    if not current_app.db_available:
        output_folder = current_app.config["OUTPUT_FOLDER"]
        # output_id may be the bare filename (with or without extension)
        fname = output_id if output_id.endswith(".jpg") else f"{output_id}.jpg"
        output_path = os.path.join(os.path.abspath(output_folder), fname)
        if not os.path.exists(output_path):
            # Try treating output_id as the full filename stored in context
            matches = [
                f for f in os.listdir(os.path.abspath(output_folder))
                if f.startswith(output_id) or output_id in f
            ]
            if not matches:
                return error_response("Output file not found", 404)
            output_path = os.path.join(os.path.abspath(output_folder), matches[0])
        return send_file(output_path, as_attachment=True,
                         download_name=f"saree-draping-{os.path.basename(output_path)}")

    # ── DB mode ────────────────────────────────────────────────────────────────
    try:
        db  = current_app.mongo.db
        doc = db.generated_outputs.find_one(
            {"_id": ObjectId(output_id), "user_id": request.user_id, "is_deleted": False}
        )
    except Exception:
        return error_response("Database unavailable", 503)
    if not doc:
        return error_response("Output not found", 404)

    output_path = doc["output_image"]
    if not os.path.exists(output_path):
        return error_response("Output file not found on server", 404)

    return send_file(output_path, as_attachment=True)


@render_bp.route("/cleanup/<output_id>", methods=["DELETE"])
@token_required
def cleanup(output_id):
    """
    Privacy cleanup: soft-delete record and remove all associated temp files.
    In no-DB mode, just deletes the output file from the outputs folder.
    """
    if not current_app.db_available:
        output_folder = os.path.abspath(current_app.config["OUTPUT_FOLDER"])
        fname = output_id if output_id.endswith(".jpg") else f"{output_id}.jpg"
        output_path = os.path.join(output_folder, fname)
        if not os.path.exists(output_path):
            matches = [f for f in os.listdir(output_folder)
                       if f.startswith(output_id) or output_id in f]
            for m in matches:
                delete_file(os.path.join(output_folder, m))
        else:
            delete_file(output_path)
        return success_response(None, "Files cleaned up successfully")

    try:
        db  = current_app.mongo.db
        doc = db.generated_outputs.find_one(
            {"_id": ObjectId(output_id), "user_id": request.user_id}
        )
    except Exception:
        return error_response("Database unavailable", 503)
    if not doc:
        return error_response("Output not found", 404)

    for field in ["body_image", "saree_image", "blouse_image", "output_image"]:
        delete_file(doc.get(field, ""))
    db.generated_outputs.update_one(
        {"_id": ObjectId(output_id)},
        {"$set": {"is_deleted": True}},
    )
    return success_response(None, "Files cleaned up successfully")


@render_bp.route("/history", methods=["GET"])
@token_required
def history():
    """Get current user's draping history."""
    db   = current_app.mongo.db
    docs = list(
        db.generated_outputs.find(
            {"user_id": request.user_id, "is_deleted": False},
            {"pose_data": 0},   # Exclude heavy pose data from list view
        ).sort("created_at", -1).limit(20)
    )
    for d in docs:
        d["_id"] = str(d["_id"])
        d["created_at"] = str(d.get("created_at"))
    return success_response(docs)
