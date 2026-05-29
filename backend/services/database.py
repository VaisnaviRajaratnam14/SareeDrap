"""
Database service — MongoDB connection management.

Supports both local MongoDB and MongoDB Atlas (cloud).
The app runs fully without a database; only auth/history routes are disabled.

Usage:
    from services.database import connect_db, get_db, check_connection
"""

import os
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError

_client: MongoClient | None = None
_db = None
_connected: bool = False


def connect_db(uri: str, db_name: str = "saree_draping_db", retries: int = 2) -> bool:
    """
    Connect to MongoDB using the given URI.
    Retries up to `retries` times before giving up.
    Returns True on success, False otherwise.
    """
    global _client, _db, _connected

    # Extract db name from URI if present (Atlas URIs include it after the host)
    # e.g. mongodb+srv://user:pass@cluster.net/saree_draping_db
    uri_db = uri.rstrip("/").rsplit("/", 1)
    if len(uri_db) == 2 and uri_db[1] and "?" not in uri_db[1]:
        db_name = uri_db[1].split("?")[0] or db_name

    print("─" * 55)
    print("  Database      : Connecting to MongoDB...")
    atlas = "mongodb+srv" in uri or "mongodb.net" in uri
    print(f"  URI type      : {'MongoDB Atlas (cloud)' if atlas else 'Local MongoDB'}")

    for attempt in range(1, retries + 2):
        try:
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=8000,
                socketTimeoutMS=10000,
            )
            client.admin.command("ping")
            _client    = client
            _db        = client[db_name]
            _connected = True
            print(f"  Database      : ✓ Connected  (db={db_name})")
            return True

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            short = str(e).split("\n")[0][:80]
            if attempt <= retries:
                print(f"  Database      : attempt {attempt} failed — {short}")
                print(f"                  retrying in 2s...")
                time.sleep(2)
            else:
                print(f"  Database      : ✗ Unavailable — {short}")

        except ConfigurationError as e:
            print(f"  Database      : ✗ Bad URI — {e}")
            break

        except Exception as e:
            print(f"  Database      : ✗ Unexpected error — {e}")
            break

    _client    = None
    _db        = None
    _connected = False
    print("  Database      : Auth/history routes disabled. Other routes still work.")
    return False


def get_db():
    """Return the active database object, or None if not connected."""
    return _db


def check_connection() -> bool:
    """Ping the server to verify the connection is still alive."""
    global _connected
    if _client is None:
        _connected = False
        return False
    try:
        _client.admin.command("ping")
        _connected = True
        return True
    except Exception:
        _connected = False
        return False


def is_connected() -> bool:
    """Fast cached check — does not send a network ping."""
    return _connected
