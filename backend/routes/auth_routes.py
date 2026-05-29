"""
Authentication routes: register, login, logout, profile.
"""

from flask import Blueprint, request, current_app
from bson import ObjectId
from models.schemas import user_schema
from utils.helpers import (
    hash_password, check_password, generate_token,
    token_required, success_response, error_response,
)

auth_bp = Blueprint("auth", __name__)


def _db_required(f):
    """Return 503 if MongoDB is not connected, with standard error shape."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_app.db_available:
            from flask import jsonify
            return jsonify({
                "success": False,
                "message": "Database unavailable",
                "error":   (
                    "MongoDB is not connected. "
                    "Set MONGO_URI in backend/.env and restart. "
                    "See README → MongoDB Atlas Setup."
                ),
            }), 503
        return f(*args, **kwargs)
    return decorated


@auth_bp.route("/register", methods=["POST"])
@_db_required
def register():
    """Register a new user."""
    data = request.get_json()
    if not data:
        return error_response("No data provided")

    name  = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    role  = data.get("role", "user")

    if not name or not email or not password:
        return error_response("Name, email, and password are required")

    db = current_app.mongo.db
    if db.users.find_one({"email": email}):
        return error_response("Email already registered", 409)

    doc = user_schema(name, email, hash_password(password), role)
    result = db.users.insert_one(doc)
    token = generate_token(str(result.inserted_id), role)

    return success_response(
        {"token": token, "user": {"id": str(result.inserted_id), "name": name, "email": email, "role": role}},
        "Registration successful",
        201,
    )


@auth_bp.route("/login", methods=["POST"])
@_db_required
def login():
    """Authenticate user and return JWT."""
    data = request.get_json()
    if not data:
        return error_response("No data provided")

    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return error_response("Email and password are required")

    db   = current_app.mongo.db
    user = db.users.find_one({"email": email, "is_active": True})
    if not user or not check_password(password, user["password_hash"]):
        return error_response("Invalid email or password", 401)

    token = generate_token(str(user["_id"]), user.get("role", "user"))
    return success_response(
        {
            "token": token,
            "user": {
                "id": str(user["_id"]),
                "name": user["name"],
                "email": user["email"],
                "role": user.get("role", "user"),
            },
        },
        "Login successful",
    )


@auth_bp.route("/profile", methods=["GET"])
@_db_required
@token_required
def profile():
    """Get current user's profile."""
    db   = current_app.mongo.db
    user = db.users.find_one({"_id": ObjectId(request.user_id)})
    if not user:
        return error_response("User not found", 404)

    return success_response(
        {
            "id": str(user["_id"]),
            "name": user["name"],
            "email": user["email"],
            "role": user.get("role", "user"),
            "created_at": str(user.get("created_at")),
        }
    )
