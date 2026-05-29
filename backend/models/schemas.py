"""
MongoDB document schemas (used as validation templates / factory functions).
PyMongo is schema-less; these dicts define the expected shape of each collection.
"""

from datetime import datetime


# ── Users Collection ───────────────────────────────────────────────────────────
def user_schema(name: str, email: str, password_hash: str, role: str = "user") -> dict:
    return {
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "role": role,              # "user" | "admin"
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "is_active": True,
        "profile_image": None,
    }


# ── Sarees Collection ──────────────────────────────────────────────────────────
def saree_schema(
    name: str,
    image_path: str,
    fabric_type: str = "silk",
    draping_style: str = "nivi",
    border_color: str = "",
    uploaded_by: str = "admin",
) -> dict:
    return {
        "name": name,
        "image_path": image_path,
        "fabric_type": fabric_type,          # silk | cotton | georgette | chiffon
        "draping_style": draping_style,      # nivi | bridal | hanging | gujarati
        "border_color": border_color,
        "dominant_color": "",
        "is_template": True,
        "uploaded_by": uploaded_by,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


# ── Blouses Collection ─────────────────────────────────────────────────────────
def blouse_schema(
    name: str,
    image_path: str,
    neck_type: str = "round",
    sleeve_type: str = "short",
    uploaded_by: str = "admin",
) -> dict:
    return {
        "name": name,
        "image_path": image_path,
        "neck_type": neck_type,       # round | boat | v_neck | square
        "sleeve_type": sleeve_type,   # sleeveless | short | half | full
        "color": "",
        "is_template": True,
        "uploaded_by": uploaded_by,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }


# ── Generated Outputs Collection ───────────────────────────────────────────────
def output_schema(
    user_id: str,
    body_image: str,
    saree_image: str,
    blouse_image: str,
    output_image: str,
    draping_style: str,
    blouse_config: dict,
) -> dict:
    return {
        "user_id": user_id,
        "body_image": body_image,
        "saree_image": saree_image,
        "blouse_image": blouse_image,
        "output_image": output_image,
        "draping_style": draping_style,
        "blouse_config": blouse_config,   # {neck_type, sleeve_type, color}
        "pose_data": {},                  # filled after pose detection
        "created_at": datetime.utcnow(),
        "is_deleted": False,              # soft-delete for privacy cleanup
    }


# ── Admin Collection ───────────────────────────────────────────────────────────
def admin_log_schema(admin_id: str, action: str, target: str, details: dict = None) -> dict:
    return {
        "admin_id": admin_id,
        "action": action,
        "target": target,
        "details": details or {},
        "timestamp": datetime.utcnow(),
    }
