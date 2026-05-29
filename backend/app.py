"""
Main Flask application entry point for the AI Saree Draping System.
Registers all blueprints and configures the app.
"""

import os
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from routes.upload_routes import upload_bp
from routes.pose_routes import pose_bp
from routes.saree_routes import saree_bp
from routes.blouse_routes import blouse_bp
from routes.render_routes import render_bp
from routes.admin_routes import admin_bp
from routes.auth_routes import auth_bp
from routes.static_routes import register_static_routes
from routes.processing import process_bp
from routes.dataset_routes import dataset_bp
from routes.train_routes import train_bp
from services.database import connect_db, get_db, is_connected

# Load environment variables
load_dotenv()

# ML weights path (checked once at startup)
_WEIGHTS_PATH = (
    Path(__file__).parent / "dataset" / "trained_models" / "saree_tryon_model.pth"
)


def _ml_engine_status() -> str:
    try:
        if _WEIGHTS_PATH.exists() and _WEIGHTS_PATH.stat().st_size > 1024:
            return "available"
    except Exception:
        pass
    return "fallback"


def create_app():
    app = Flask(__name__)

    # ── Configuration ──────────────────────────────────────────────────────────
    app.config["SECRET_KEY"]         = os.getenv("SECRET_KEY", "saree-drape-secret-key")
    app.config["MONGO_URI"]          = os.getenv(
        "MONGO_URI", "mongodb://localhost:27017/saree_draping_db"
    )
    app.config["UPLOAD_FOLDER"]      = os.getenv("UPLOAD_FOLDER", "uploads")
    app.config["OUTPUT_FOLDER"]      = os.getenv("OUTPUT_FOLDER", "outputs")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

    # ── CORS ───────────────────────────────────────────────────────────────────
    CORS(app, origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        os.getenv("ALLOWED_ORIGINS", ""),
    ])

    # ── Startup banner ─────────────────────────────────────────────────────────
    print("═" * 55)
    print("  SareeDrape Studio — AI Saree Draping System")
    print("═" * 55)
    print(f"  Backend       : starting on port 5000")

    # ── MongoDB (via database service) ─────────────────────────────────────────
    db_ok = connect_db(app.config["MONGO_URI"])
    app.db_available = db_ok

    # Attach a thin proxy so existing routes using app.mongo.db keep working
    if db_ok:
        class _MongoProxy:
            @property
            def db(self):
                return get_db()
        app.mongo = _MongoProxy()
    else:
        app.mongo = None

    # ── ML engine status ───────────────────────────────────────────────────────
    ml_status = _ml_engine_status()
    print(f"  ML engine     : {ml_status} "
          f"({'weights found' if ml_status == 'available' else 'geometric fallback — no weights'})")

    # ── Ensure required directories exist ──────────────────────────────────────
    for folder in [
        app.config["UPLOAD_FOLDER"],
        app.config["OUTPUT_FOLDER"],
        "uploads/body",
        "uploads/saree",
        "uploads/blouse",
        "models",
        "dataset",
        "dataset/raw",
        "dataset/raw/body_images",
        "dataset/raw/saree_images",
        "dataset/raw/blouse_materials",
        "dataset/cleaned",
        "dataset/cleaned/body_images",
        "dataset/cleaned/saree_images",
        "dataset/cleaned/blouse_materials",
        "dataset/cleaned/masks",
        "dataset/annotations",
        "dataset/trained_models",
        "dataset/trained_models/checkpoints",
        "outputs",
    ]:
        os.makedirs(folder, exist_ok=True)

    # ── Register Blueprints ────────────────────────────────────────────────────
    app.register_blueprint(auth_bp,    url_prefix="/api/auth")
    app.register_blueprint(upload_bp,  url_prefix="/api/upload")
    app.register_blueprint(pose_bp,    url_prefix="/api/pose")
    app.register_blueprint(saree_bp,   url_prefix="/api/saree")
    app.register_blueprint(blouse_bp,  url_prefix="/api/blouse")
    app.register_blueprint(render_bp,  url_prefix="/api/render")
    app.register_blueprint(admin_bp,   url_prefix="/api/admin")
    app.register_blueprint(process_bp,  url_prefix="/api/process")
    app.register_blueprint(dataset_bp,  url_prefix="/api/dataset")
    app.register_blueprint(train_bp,    url_prefix="/api/train")

    # ── Static file serving (outputs + uploads) ────────────────────────────────
    register_static_routes(app)

    # ── Health route ───────────────────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        db_live = is_connected()
        return jsonify({
            "backend":   "running",
            "database":  "connected" if db_live else "disconnected",
            "ml_engine": _ml_engine_status(),
        }), 200

    # ── Final startup summary ──────────────────────────────────────────────────
    print("─" * 55)
    print(f"  Auth/history  : {'enabled' if db_ok else 'disabled (no DB)'}")
    print(f"  AI processing : enabled (always)")
    print(f"  Health check  : GET /api/health")
    print("═" * 55)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=os.getenv("FLASK_DEBUG", "true").lower() == "true", port=5000)
