"""
Upload routes: handle body image, saree image, and blouse image uploads.
"""

import os
from flask import Blueprint, request, current_app, send_file
from utils.helpers import save_upload, success_response, error_response, token_required

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/body", methods=["POST"])
@token_required
def upload_body():
    """Upload user body image."""
    if "image" not in request.files:
        return error_response("No image file provided")
    file = request.files["image"]
    if file.filename == "":
        return error_response("No file selected")
    try:
        path = save_upload(file, "body")
        return success_response(
            {"file_path": path, "filename": os.path.basename(path)},
            "Body image uploaded successfully",
            201,
        )
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Upload failed: {str(e)}", 500)


@upload_bp.route("/saree", methods=["POST"])
@token_required
def upload_saree():
    """Upload saree image/material."""
    if "image" not in request.files:
        return error_response("No image file provided")
    file = request.files["image"]
    if file.filename == "":
        return error_response("No file selected")
    try:
        path = save_upload(file, "saree")
        return success_response(
            {"file_path": path, "filename": os.path.basename(path)},
            "Saree image uploaded successfully",
            201,
        )
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Upload failed: {str(e)}", 500)


@upload_bp.route("/blouse", methods=["POST"])
@token_required
def upload_blouse():
    """Upload blouse material image."""
    if "image" not in request.files:
        return error_response("No image file provided")
    file = request.files["image"]
    if file.filename == "":
        return error_response("No file selected")
    try:
        path = save_upload(file, "blouse")
        return success_response(
            {"file_path": path, "filename": os.path.basename(path)},
            "Blouse image uploaded successfully",
            201,
        )
    except ValueError as e:
        return error_response(str(e))
    except Exception as e:
        return error_response(f"Upload failed: {str(e)}", 500)


@upload_bp.route("/file/<path:filename>", methods=["GET"])
def serve_file(filename):
    """Serve an uploaded file by its relative path."""
    base = current_app.config["UPLOAD_FOLDER"]
    full_path = os.path.join(base, filename)
    if not os.path.exists(full_path):
        return error_response("File not found", 404)
    return send_file(full_path)
