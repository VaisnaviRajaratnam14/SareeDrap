"""
Shared utility helpers: file saving, JWT auth, response formatting.
"""

import os
import uuid
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import current_app, request, jsonify
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


# ── File Utilities ─────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    """Check if the file extension is permitted."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file, subfolder: str) -> str:
    """
    Save an uploaded FileStorage object under uploads/<subfolder>.
    Returns the relative path string.
    """
    if not allowed_file(file.filename):
        raise ValueError(f"File type not allowed: {file.filename}")

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], subfolder)
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, unique_name)
    file.save(full_path)
    return full_path.replace("\\", "/")  # forward-slashes for JSON safety


def delete_file(path: str) -> bool:
    """Safely delete a file; returns True if deleted."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            return True
    except OSError:
        pass
    return False


# ── Password Utilities ─────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT Utilities ──────────────────────────────────────────────────────────────
def generate_token(user_id: str, role: str = "user") -> str:
    """Generate a signed JWT token."""
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=int(os.getenv("JWT_EXPIRY_HOURS", 24))),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, current_app.config["SECRET_KEY"], algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])


def token_required(f):
    """Decorator: protect routes with JWT auth.
    When DB is unavailable auth is bypassed so AI processing still works.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Bypass auth entirely when DB is unavailable
        if not current_app.db_available:
            request.user_id   = "anonymous"
            request.user_role = "user"
            return f(*args, **kwargs)

        token = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            data = decode_token(token)
            request.user_id = data["user_id"]
            request.user_role = data["role"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator: restrict to admin users only."""
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        if request.user_role != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated


# ── Response Helpers ───────────────────────────────────────────────────────────
def success_response(data=None, message="Success", status=200):
    return jsonify({"success": True, "message": message, "data": data}), status


def error_response(message="An error occurred", status=400, extra=None):
    body = {"success": False, "error": message}
    if extra:
        for k, v in extra.items():
            if k not in ("success", "error"):
                body[k] = v
    return jsonify(body), status
