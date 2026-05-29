"""
Database Seed Script
====================
Run once to populate MongoDB with:
 - Admin user
 - Sample saree templates
 - Sample blouse templates

Usage:
    python seed.py
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import bcrypt

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/saree_draping_db")
client    = MongoClient(MONGO_URI)
db        = client.get_default_database()


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def seed_users():
    db.users.delete_many({})
    users = [
        {
            "name": "Admin User",
            "email": "admin@saree.com",
            "password_hash": hash_pw("admin123"),
            "role": "admin",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "profile_image": None,
        },
        {
            "name": "Demo User",
            "email": "demo@saree.com",
            "password_hash": hash_pw("demo123"),
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "profile_image": None,
        },
    ]
    db.users.insert_many(users)
    print(f"✓ Seeded {len(users)} users")


def seed_sarees():
    db.sarees.delete_many({})
    sarees = [
        {"name": "Kanjivaram Silk",     "fabric_type": "silk",      "draping_style": "nivi",     "border_color": "#8B0000", "dominant_color": "#C41FFF", "is_template": True, "image_path": "dataset/saree_samples/kanjivaram.jpg"},
        {"name": "Banarasi Silk",       "fabric_type": "silk",      "draping_style": "bridal",   "border_color": "#DAA520", "dominant_color": "#8B0057", "is_template": True, "image_path": "dataset/saree_samples/banarasi.jpg"},
        {"name": "Chiffon Pastel",      "fabric_type": "chiffon",   "draping_style": "hanging",  "border_color": "#FFB6C1", "dominant_color": "#E8A0BF", "is_template": True, "image_path": "dataset/saree_samples/chiffon.jpg"},
        {"name": "Cotton Tant",         "fabric_type": "cotton",    "draping_style": "nivi",     "border_color": "#228B22", "dominant_color": "#FFFFFF", "is_template": True, "image_path": "dataset/saree_samples/tant.jpg"},
        {"name": "Georgette Party",     "fabric_type": "georgette", "draping_style": "gujarati", "border_color": "#4B0082", "dominant_color": "#9400D3", "is_template": True, "image_path": "dataset/saree_samples/georgette.jpg"},
        {"name": "Bridal Red Silk",     "fabric_type": "silk",      "draping_style": "bridal",   "border_color": "#B8860B", "dominant_color": "#CC0000", "is_template": True, "image_path": "dataset/saree_samples/bridal_red.jpg"},
    ]
    now = datetime.utcnow()
    for s in sarees:
        s["uploaded_by"] = "admin"
        s["created_at"]  = now
        s["updated_at"]  = now
    db.sarees.insert_many(sarees)
    print(f"✓ Seeded {len(sarees)} saree templates")


def seed_blouses():
    db.blouses.delete_many({})
    blouses = [
        {"name": "Classic Round Neck",   "neck_type": "round",      "sleeve_type": "short",      "color": "#8B0057", "is_template": True, "image_path": "dataset/blouse_samples/round_short.jpg"},
        {"name": "Elegant Boat Neck",    "neck_type": "boat",       "sleeve_type": "sleeveless", "color": "#DAA520", "is_template": True, "image_path": "dataset/blouse_samples/boat_sl.jpg"},
        {"name": "V-Neck Half Sleeve",   "neck_type": "v_neck",     "sleeve_type": "half",       "color": "#4B0082", "is_template": True, "image_path": "dataset/blouse_samples/vneck_half.jpg"},
        {"name": "Full Sleeve Bridal",   "neck_type": "sweetheart", "sleeve_type": "full",       "color": "#CC0000", "is_template": True, "image_path": "dataset/blouse_samples/sweetheart_full.jpg"},
        {"name": "Square Neck Casual",   "neck_type": "square",     "sleeve_type": "short",      "color": "#228B22", "is_template": True, "image_path": "dataset/blouse_samples/square_short.jpg"},
    ]
    now = datetime.utcnow()
    for b in blouses:
        b["uploaded_by"] = "admin"
        b["created_at"]  = now
        b["updated_at"]  = now
    db.blouses.insert_many(blouses)
    print(f"✓ Seeded {len(blouses)} blouse templates")


def create_indexes():
    db.users.create_index("email", unique=True)
    db.generated_outputs.create_index("user_id")
    db.generated_outputs.create_index("created_at")
    print("✓ Indexes created")


if __name__ == "__main__":
    print("Seeding database...")
    seed_users()
    seed_sarees()
    seed_blouses()
    create_indexes()
    print("\n✅ Database seeded successfully!")
    print("   Admin login: admin@saree.com / admin123")
    print("   Demo login:  demo@saree.com  / demo123")
