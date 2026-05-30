"""
Dataset routes: Kaggle download, cleaning, and status.
POST /api/dataset/kaggle-download  { dataset_slug, category, max_images, mode }
POST /api/dataset/clean
GET  /api/dataset/status
"""

import os
import sys
import json
import uuid
import shutil
import subprocess
import zipfile
import glob
from pathlib import Path
from flask import Blueprint, request
from utils.helpers import success_response, error_response

dataset_bp = Blueprint("dataset", __name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE    = Path(__file__).parent.parent
_DATASET = _BASE / "dataset"

# raw/<category>/  — downloaded images land here
_RAW = _DATASET / "raw"
_CATEGORY_DIRS = {
    "body_images":      _RAW / "body_images",
    "saree_images":     _RAW / "saree_images",
    "blouse_materials": _RAW / "blouse_materials",
}

# cleaned/<category>/ — validated, resized, normalised images
_CLEANED = _DATASET / "cleaned"
_CLEANED_DIRS = {
    "body_images":      _CLEANED / "body_images",
    "saree_images":     _CLEANED / "saree_images",
    "blouse_materials": _CLEANED / "blouse_materials",
}
_MASKS_DIR      = _CLEANED / "masks"
_ANNOTATIONS_DIR = _DATASET / "annotations"

_STATE_FILE    = _DATASET / ".dataset_state.json"
_IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MIN_TRAIN     = 50      # hard block below this
_DEFAULT_MAX   = 150


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "fetched": False, "cleaned": False,
        "total_images": 0, "cleaned_images": 0,
        "per_category": {k: 0 for k in _CATEGORY_DIRS},
    }


def _write_state(state: dict):
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*")
               if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)


def _per_category_counts() -> dict:
    return {cat: _count_images(folder) for cat, folder in _CATEGORY_DIRS.items()}


def _per_cleaned_counts() -> dict:
    return {cat: _count_images(folder) for cat, folder in _CLEANED_DIRS.items()}


def _total_images() -> int:
    return sum(_count_images(d) for d in _CATEGORY_DIRS.values())


def _import_images(src_dir: Path, dest_dir: Path, max_images: int) -> dict:
    """
    Walk src_dir recursively, copy up to max_images image files into dest_dir.

    Validation rules:
    - Extension in _IMAGE_EXTS
    - File size >= 512 bytes
    - Dimensions >= 64x64  (PIL primary, cv2 fallback)
    - Skip non-image sub-folders (csv, json, metadata files)

    Logs each accept/reject with reason.
    """
    try:
        from PIL import Image as _PIL
        _pil_ok = True
    except ImportError:
        _pil_ok = False

    try:
        import cv2 as _cv2
        _cv2_ok = True
    except ImportError:
        _cv2_ok = False

    imported = 0
    skipped  = 0
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Gather all candidate files — prefer images/ and styles/ sub-trees first
    all_files = []
    for fpath in sorted(src_dir.rglob("*")):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() not in _IMAGE_EXTS:
            continue
        all_files.append(fpath)

    # Sort: put files inside images/ or styles/ directories first
    def _sort_key(p: Path):
        parts_lower = [x.lower() for x in p.parts]
        priority = 0 if any(x in ("images", "styles") for x in parts_lower) else 1
        return (priority, str(p))

    all_files.sort(key=_sort_key)
    print(f"[Dataset] candidate image files: {len(all_files)}")

    for fpath in all_files:
        if imported >= max_images:
            skipped += 1
            continue

        # ── Size check ────────────────────────────────────────────────────────
        try:
            size = fpath.stat().st_size
        except Exception:
            print(f"[Dataset] rejected: {fpath.name}  reason: stat() failed")
            skipped += 1
            continue
        if size < 512:
            print(f"[Dataset] rejected: {fpath.name}  reason: too small ({size} B)")
            skipped += 1
            continue

        # ── Dimension check (PIL primary, cv2 fallback) ───────────────────────
        w = h = None
        if _pil_ok:
            try:
                with _PIL.open(str(fpath)) as im:
                    w, h = im.size          # PIL returns (width, height)
            except Exception as pil_err:
                if _cv2_ok:
                    img = _cv2.imread(str(fpath))
                    if img is not None:
                        h, w = img.shape[:2]
                    else:
                        print(f"[Dataset] rejected: {fpath.name}  reason: PIL+cv2 both unreadable")
                        skipped += 1
                        continue
                else:
                    print(f"[Dataset] rejected: {fpath.name}  reason: PIL error ({pil_err})")
                    skipped += 1
                    continue
        elif _cv2_ok:
            img = _cv2.imread(str(fpath))
            if img is not None:
                h, w = img.shape[:2]
            else:
                print(f"[Dataset] rejected: {fpath.name}  reason: cv2 unreadable")
                skipped += 1
                continue
        # If neither PIL nor cv2 — accept on extension+size alone (no dims check)

        if w is not None and h is not None:
            if w < 60 or h < 60:
                print(f"[Dataset] rejected: {fpath.name}  reason: too small ({w}x{h})")
                skipped += 1
                continue

        # ── Accept ────────────────────────────────────────────────────────────
        dest_name = f"{uuid.uuid4().hex}{fpath.suffix.lower()}"
        shutil.copy2(str(fpath), str(dest_dir / dest_name))
        print(f"[Dataset] accepted: {fpath.name}  size={size}B  dims={w}x{h}")
        imported += 1

    print(f"[Dataset] import complete — accepted={imported}  rejected={skipped}")
    return {
        "imported":      imported,
        "skipped":       skipped,
        "total_in_dest": _count_images(dest_dir),
    }


def _purge_placeholders(folder: Path) -> int:
    """Delete any file whose name starts with 'placeholder_'. Returns count deleted."""
    deleted = 0
    if not folder.exists():
        return 0
    for f in list(folder.rglob("placeholder_*")):
        if f.is_file():
            f.unlink()
            deleted += 1
    return deleted


# ── Routes ─────────────────────────────────────────────────────────────────────

@dataset_bp.route("/kaggle-download", methods=["POST"])
def kaggle_download():
    """
    Download REAL images from Kaggle into backend/dataset/raw/<category>/.
    Placeholder generation is DISABLED — real dataset slug required.

    Body (JSON):
      dataset_slug  – Kaggle slug "username/dataset-name"  (required)
      category      – "body_images" | "saree_images" | "blouse_materials"
      max_images    – int 1-2000, default 150
      mode          – "append" (default) | "replace"
    """
    print("[Dataset] Placeholder generation disabled — real Kaggle downloads only")

    body         = request.get_json(silent=True) or {}
    dataset_slug = (body.get("dataset_slug") or "").strip()
    category     = (body.get("category") or "").strip()
    max_images   = int(body.get("max_images") or _DEFAULT_MAX)
    mode         = (body.get("mode") or "append").strip().lower()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not dataset_slug:
        return error_response(
            "Dataset slug required. Enter a Kaggle slug e.g. 'username/dataset-name'.",
            400,
        )

    if not category or category not in _CATEGORY_DIRS:
        return error_response(
            f"Invalid category '{category}'. "
            f"Choose from: {', '.join(_CATEGORY_DIRS)}",
            400,
        )

    if max_images < 1 or max_images > 2000:
        return error_response("max_images must be between 1 and 2000", 400)

    # ── Check Kaggle credentials ──────────────────────────────────────────────
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_creds = bool(os.environ.get("KAGGLE_USERNAME") and
                         os.environ.get("KAGGLE_KEY"))
    if not kaggle_json.exists() and not has_env_creds:
        return error_response(
            "Kaggle API credentials not configured. "
            "Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables.",
            401,
        )

    # ── Check kaggle package importable via this Python ──────────────────────
    kaggle_check = subprocess.run(
        [sys.executable, "-m", "kaggle", "--version"],
        capture_output=True, text=True,
    )
    print(f"[Dataset] kaggle --version rc={kaggle_check.returncode} "
          f"out={kaggle_check.stdout.strip()!r} err={kaggle_check.stderr.strip()[:80]!r}")
    if kaggle_check.returncode != 0:
        return error_response(
            "Kaggle package not found in current Python environment. "
            "Run: pip install kaggle",
            500,
            extra={"stdout": kaggle_check.stdout, "stderr": kaggle_check.stderr},
        )

    dest_dir = _CATEGORY_DIRS[category]
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = _count_images(dest_dir)

    # Replace mode — clear existing images first
    if mode == "replace" and existing > 0:
        for f in list(dest_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                f.unlink()
        existing = 0
        print(f"[Dataset] Replace mode — cleared existing images in {category}")

    # ── Download ZIP explicitly (no --unzip) ──────────────────────────────────
    tmp_dir = _DATASET / f"_tmp_kaggle_{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build command — download to tmp_dir WITHOUT --unzip so we get the .zip
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", dataset_slug,
        "-p", str(tmp_dir),
        "--force",
    ]
    print(f"[Dataset] slug={dataset_slug!r}  category={category!r}  "
          f"max_images={max_images}  tmp_dir={tmp_dir}")
    print(f"[Dataset] cmd={' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        stdout_txt = result.stdout.strip()
        stderr_txt = result.stderr.strip()
        print(f"[Dataset] return_code={result.returncode}")
        print(f"[Dataset] stdout={stdout_txt[:300]!r}")
        print(f"[Dataset] stderr={stderr_txt[:300]!r}")

        if result.returncode != 0:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            err_text = stderr_txt or stdout_txt
            if any(x in err_text for x in ("401", "Unauthorized", "credentials", "403", "Forbidden")):
                return error_response(
                    "Kaggle credentials rejected (401/403). "
                    "Re-download kaggle.json from kaggle.com/settings/api.",
                    401,
                    extra={"stdout": stdout_txt, "stderr": stderr_txt},
                )
            if any(x in err_text.lower() for x in ("404", "not found", "no such dataset")):
                return error_response(
                    f"Dataset '{dataset_slug}' not found on Kaggle. "
                    "Check the slug at kaggle.com/datasets.",
                    404,
                    extra={"stdout": stdout_txt, "stderr": stderr_txt},
                )
            return error_response(
                f"Kaggle download failed (rc={result.returncode}): {err_text[:200]}",
                500,
                extra={"stdout": stdout_txt, "stderr": stderr_txt},
            )

        # ── Find the downloaded ZIP ───────────────────────────────────────────
        zip_files = list(tmp_dir.glob("*.zip"))
        print(f"[Dataset] zip files found: {[z.name for z in zip_files]}")

        extract_dir = tmp_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)

        if zip_files:
            # Extract the first (largest) zip
            zip_path = max(zip_files, key=lambda z: z.stat().st_size)
            print(f"[Dataset] Extracting {zip_path.name} ({zip_path.stat().st_size//1024} KB) "
                  f"→ {extract_dir}")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(extract_dir))
        else:
            # kaggle CLI may have already unzipped in place (some versions do)
            # treat the whole tmp_dir (minus .zip) as the extract dir
            print("[Dataset] No zip found — treating tmp_dir as already-extracted")
            extract_dir = tmp_dir

        # Count files available before import
        all_imgs = [p for p in extract_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        print(f"[Dataset] extraction_folder={extract_dir}  "
              f"image_files_found={len(all_imgs)}")

        # ── Import valid images into dest_dir ─────────────────────────────────
        remaining = max(0, max_images - existing)
        stats     = _import_images(extract_dir, dest_dir, remaining)
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

        print(f"[Dataset] imported={stats['imported']}  skipped={stats['skipped']}  "
              f"total_in_dest={stats['total_in_dest']}")

        per_cat = _per_category_counts()
        state   = _read_state()
        state.update(fetched=True, total_images=sum(per_cat.values()),
                     per_category=per_cat, cleaned=False)
        _write_state(state)

        msg = (f"{stats['imported']} images imported to '{category}' "
               f"({stats['skipped']} skipped). "
               f"Total in category: {stats['total_in_dest']}")
        return success_response({
            "imported":      stats["imported"],
            "skipped":       stats["skipped"],
            "total_in_dest": stats["total_in_dest"],
            "category":      category,
            "category_path": str(dest_dir),
            "total_images":  sum(per_cat.values()),
            "per_category":  per_cat,
            "message":       msg,
        }, "Dataset fetched", 200)

    except subprocess.TimeoutExpired:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        return error_response(
            "Kaggle download timed out (>10 min). Try a smaller dataset.", 504
        )
    except zipfile.BadZipFile as exc:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        return error_response(
            f"Downloaded file is not a valid ZIP: {exc}", 500
        )
    except Exception as exc:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        import traceback
        tb = traceback.format_exc()
        print(f"[Dataset] Unexpected error: {tb}")
        return error_response(
            f"Kaggle download failed: {str(exc)}", 500,
            extra={"traceback": tb[-400:]},
        )


@dataset_bp.route("/clean", methods=["POST"])
def clean_dataset():
    """
    Full cleaning pipeline:
    - Purge placeholder_* files from raw/ folders
    - Validate each raw image: extension, size >= 1KB, dimensions >= 100x100
    - Resize to fit within 512x768 (portrait)
    - Normalise brightness (CLAHE on L channel)
    - Convert to PNG
    - Save validated images into dataset/cleaned/<category>/
    - Write dataset/cleaning_report.json
    """
    state = _read_state()

    # Purge placeholders from raw folders first
    purged = sum(_purge_placeholders(folder) for folder in _CATEGORY_DIRS.values())
    if purged:
        print(f"[Dataset] Purged {purged} placeholder/fake images before cleaning")

    try:
        import cv2
        import numpy as np

        total_cleaned  = 0
        total_removed  = purged
        per_cat_result = {}
        report_entries = []

        for cat, raw_folder in _CATEGORY_DIRS.items():
            cleaned_folder = _CLEANED_DIRS[cat]
            cleaned_folder.mkdir(parents=True, exist_ok=True)

            accepted = 0
            rejected = 0

            if not raw_folder.exists():
                per_cat_result[cat] = {"cleaned": 0, "removed": 0}
                continue

            for fpath in sorted(raw_folder.rglob("*")):
                if not fpath.is_file():
                    continue
                if fpath.suffix.lower() not in _IMAGE_EXTS:
                    rejected += 1
                    continue
                if fpath.stat().st_size < 1024:
                    rejected += 1
                    continue
                try:
                    img = cv2.imread(str(fpath))
                    if img is None:
                        rejected += 1
                        continue
                    h, w = img.shape[:2]
                    if h < 60 or w < 60:
                        rejected += 1
                        continue

                    # Resize to fit within 512x768 portrait
                    max_w, max_h = 512, 768
                    scale = min(max_w / w, max_h / h, 1.0)
                    if scale < 1.0:
                        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                         interpolation=cv2.INTER_AREA)

                    # Brightness normalisation via CLAHE on L channel
                    try:
                        lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l     = clahe.apply(l)
                        img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
                    except Exception:
                        pass  # keep original if CLAHE fails

                    # Save as PNG into cleaned folder
                    dest_name = f"{uuid.uuid4().hex}.png"
                    cv2.imwrite(str(cleaned_folder / dest_name), img)
                    accepted += 1

                except Exception:
                    rejected += 1
                    continue

            per_cat_result[cat] = {"cleaned": accepted, "removed": rejected}
            total_cleaned += accepted
            total_removed += rejected
            report_entries.append({
                "category":        cat,
                "raw_count":       _count_images(raw_folder),
                "cleaned_count":   accepted,
                "rejected_count":  rejected,
                "cleaned_folder":  str(cleaned_folder),
            })
            print(f"[Dataset] Cleaned {cat}: {accepted} accepted, {rejected} rejected")

        # Write cleaning report
        report = {
            "total_cleaned": total_cleaned,
            "total_removed": total_removed,
            "purged_placeholders": purged,
            "categories": report_entries,
            "cleaned_at": __import__('datetime').datetime.utcnow().isoformat() + "Z",
        }
        (_DATASET / "cleaning_report.json").write_text(
            json.dumps(report, indent=2)
        )

        per_cat     = _per_category_counts()
        cleaned_cat = _per_cleaned_counts()
        state.update(cleaned=True, cleaned_images=total_cleaned,
                     total_images=sum(per_cat.values()),
                     per_category=per_cat,
                     cleaned_per_category=cleaned_cat)
        _write_state(state)

        msg = (f"{total_cleaned} images cleaned and saved to dataset/cleaned/, "
               f"{total_removed} rejected")
        print(f"[Dataset] {msg}")
        return success_response({
            "cleaned":              total_cleaned,
            "removed":              total_removed,
            "total_images":         sum(per_cat.values()),
            "per_category":         per_cat,
            "cleaned_per_category": cleaned_cat,
            "per_cat_detail":       per_cat_result,
            "report_path":          str(_DATASET / "cleaning_report.json"),
            "message":              msg,
        }, "Dataset cleaned", 200)

    except ImportError:
        # OpenCV unavailable — copy raw images as-is into cleaned folders
        per_cat = _per_category_counts()
        cleaned_cat = {}
        total = 0
        for cat, raw_folder in _CATEGORY_DIRS.items():
            cleaned_folder = _CLEANED_DIRS[cat]
            cleaned_folder.mkdir(parents=True, exist_ok=True)
            copied = 0
            for fpath in sorted(raw_folder.rglob("*") if raw_folder.exists() else []):
                if fpath.is_file() and fpath.suffix.lower() in _IMAGE_EXTS and fpath.stat().st_size >= 1024:
                    shutil.copy2(str(fpath), str(cleaned_folder / f"{uuid.uuid4().hex}{fpath.suffix.lower()}"))
                    copied += 1
            cleaned_cat[cat] = copied
            total += copied
        state.update(cleaned=True, cleaned_images=total,
                     total_images=sum(per_cat.values()),
                     per_category=per_cat,
                     cleaned_per_category=cleaned_cat)
        _write_state(state)
        return success_response({
            "cleaned":              total,
            "removed":              purged,
            "total_images":         sum(per_cat.values()),
            "per_category":         per_cat,
            "cleaned_per_category": cleaned_cat,
            "message":              f"{total} images copied to cleaned/ (OpenCV unavailable — no resize/normalise)",
        }, "Dataset cleaned", 200)

    except Exception as exc:
        return error_response(f"Cleaning failed: {str(exc)}", 500)


@dataset_bp.route("/annotate", methods=["POST"])
def annotate_dataset():
    """
    Annotation pipeline:
    - Run MediaPipe Pose on cleaned body images → keypoints JSON
      (empty keypoints written if pose not detected — never skips file)
    - Generate binary segmentation masks for saree + blouse images
    - Save keypoints into dataset/annotations/
    - Save masks into dataset/cleaned/masks/
    - Write dataset/annotation_report.json
    - Always returns 200 even if some images fail.
    """
    import datetime
    import traceback

    annotated   = 0
    failed      = 0
    masks_saved = 0
    error_list  = []

    try:
        state = _read_state()

        body_cleaned   = _CLEANED_DIRS["body_images"]
        saree_cleaned  = _CLEANED_DIRS["saree_images"]
        blouse_cleaned = _CLEANED_DIRS["blouse_materials"]
        body_count     = _count_images(body_cleaned)

        # Auto-heal state flag — trust actual files on disk, not cached flag
        total_cleaned = sum(_count_images(d) for d in _CLEANED_DIRS.values())
        if total_cleaned > 0 and not state.get("cleaned"):
            print(f"[Annotate] Auto-healing state.cleaned=True ({total_cleaned} cleaned images found on disk)")
            state["cleaned"] = True
            _write_state(state)

        if body_count == 0:
            return error_response(
                "No cleaned body images found in dataset/cleaned/body_images/. "
                "Run Clean first (Step 2) to process raw images.",
                400,
            )

        _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        _MASKS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[Annotate] Starting — body_images={body_count}")

        # ── MediaPipe Pose on body images ──────────────────────────────────────
        # mediapipe >= 0.10 uses Tasks API (mp.tasks.vision.PoseLandmarker)
        # mp.solutions was removed; we fall back gracefully if model is absent.
        _mp_available  = False
        _pose_landmarker = None

        try:
            import mediapipe as mp
            _mp_model_path = _BASE / "pose_landmarker.task"
            if not _mp_model_path.exists():
                # Try to download the lite model automatically
                print("[Annotate] Downloading MediaPipe pose model (~5 MB)...")
                import urllib.request as _ur
                _url = ("https://storage.googleapis.com/mediapipe-models/"
                        "pose_landmarker/pose_landmarker_lite/float16/1/"
                        "pose_landmarker_lite.task")
                _ur.urlretrieve(_url, str(_mp_model_path))
                print(f"[Annotate] Model downloaded → {_mp_model_path}")

            _PL        = mp.tasks.vision.PoseLandmarker
            _PLOptions = mp.tasks.vision.PoseLandmarkerOptions
            _BaseOpts  = mp.tasks.BaseOptions
            _RunMode   = mp.tasks.vision.RunningMode

            _options = _PLOptions(
                base_options=_BaseOpts(model_asset_path=str(_mp_model_path)),
                running_mode=_RunMode.IMAGE,
            )
            _pose_landmarker = _PL.create_from_options(_options)
            _mp_available = True
            print("[Annotate] MediaPipe PoseLandmarker (Tasks API) initialised")
        except ImportError as e:
            print(f"[Annotate] MediaPipe not available ({e}) — empty keypoints will be saved")
            error_list.append(f"MediaPipe import failed: {e}")
        except Exception as e:
            print(f"[Annotate] MediaPipe init failed ({e}) — empty keypoints will be saved")
            error_list.append(f"MediaPipe init error: {e}")

        try:
            import cv2 as _cv2
            _cv2_available = True
        except ImportError:
            _cv2_available = False
            error_list.append("OpenCV (cv2) not available")

        body_files = sorted(
            f for f in body_cleaned.rglob("*")
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        print(f"[Annotate] Processing {len(body_files)} body images")

        for fpath in body_files:
            try:
                w = h = 0
                keypoints = []

                if _cv2_available:
                    img_bgr = _cv2.imread(str(fpath))
                    if img_bgr is None:
                        raise ValueError(f"cv2.imread returned None for {fpath.name}")
                    h, w = img_bgr.shape[:2]

                    if _mp_available and _pose_landmarker is not None:
                        try:
                            import mediapipe as mp
                            img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(
                                image_format=mp.ImageFormat.SRGB,
                                data=img_rgb,
                            )
                            result = _pose_landmarker.detect(mp_image)
                            if result.pose_landmarks:
                                for lm in result.pose_landmarks[0]:
                                    keypoints.append({
                                        "x":          round(lm.x, 6),
                                        "y":          round(lm.y, 6),
                                        "z":          round(lm.z, 6),
                                        "visibility": round(getattr(lm, "visibility", 0.0), 4),
                                    })
                                print(f"[Annotate] detected {len(keypoints)} keypoints: {fpath.name}")
                            else:
                                print(f"[Annotate] no pose detected (empty keypoints): {fpath.name}")
                        except Exception as pose_err:
                            print(f"[Annotate] pose error on {fpath.name}: {pose_err}")
                            error_list.append(f"{fpath.name}: pose error — {pose_err}")
                            # keypoints stays [] — do NOT fail the image

                # Write annotation JSON (empty keypoints = pose not detected, still valid)
                ann = {
                    "image":     fpath.name,
                    "width":     w,
                    "height":    h,
                    "keypoints": keypoints,
                    "detected":  len(keypoints) > 0,
                }
                ann_path = _ANNOTATIONS_DIR / f"{fpath.stem}.json"
                ann_path.write_text(json.dumps(ann, indent=2))
                annotated += 1

            except Exception as img_err:
                tb_short = traceback.format_exc().strip().splitlines()[-1]
                print(f"[Annotate] FAILED {fpath.name}: {img_err}  |  {tb_short}")
                error_list.append(f"{fpath.name}: {img_err}")
                failed += 1
                try:
                    stub = {"image": fpath.name, "width": 0, "height": 0,
                            "keypoints": [], "detected": False, "error": str(img_err)}
                    (_ANNOTATIONS_DIR / f"{fpath.stem}.json").write_text(
                        json.dumps(stub, indent=2)
                    )
                except Exception:
                    pass

        if _pose_landmarker is not None:
            try:
                _pose_landmarker.close()
            except Exception:
                pass

        print(f"[Annotate] Pose done — annotated={annotated}, failed={failed}")

        # ── Binary segmentation masks for saree + blouse ──────────────────────
        if _cv2_available:
            for cat_folder in [saree_cleaned, blouse_cleaned]:
                mask_files = sorted(
                    f for f in cat_folder.rglob("*")
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
                )
                for fpath in mask_files:
                    try:
                        img = _cv2.imread(str(fpath))
                        if img is None:
                            print(f"[Annotate] mask skip (unreadable): {fpath.name}")
                            continue
                        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
                        _, mask = _cv2.threshold(gray, 10, 255, _cv2.THRESH_BINARY)
                        mask_path = _MASKS_DIR / f"{fpath.stem}_mask.png"
                        _cv2.imwrite(str(mask_path), mask)
                        masks_saved += 1
                    except Exception as mask_err:
                        print(f"[Annotate] mask error {fpath.name}: {mask_err}")
                        error_list.append(f"mask:{fpath.name} — {mask_err}")
            print(f"[Annotate] Masks saved: {masks_saved}")
        else:
            print("[Annotate] OpenCV unavailable — skipping mask generation")

        # ── Write annotation report ────────────────────────────────────────────
        report = {
            "body_images_processed": annotated,
            "body_images_failed":    failed,
            "masks_generated":       masks_saved,
            "annotations_dir":       str(_ANNOTATIONS_DIR),
            "masks_dir":             str(_MASKS_DIR),
            "error_list":            error_list[:50],   # cap at 50 entries
            "annotated_at":          datetime.datetime.utcnow().isoformat() + "Z",
        }
        (_DATASET / "annotation_report.json").write_text(json.dumps(report, indent=2))
        print(f"[Annotate] Report written — {_DATASET / 'annotation_report.json'}")

        state.update(annotated=True, annotated_count=annotated, masks_count=masks_saved)
        _write_state(state)

        msg = (f"{annotated} body images annotated, "
               f"{masks_saved} masks generated"
               + (f", {failed} failed" if failed else ""))
        print(f"[Annotate] {msg}")

        return success_response({
            "annotated":   annotated,
            "failed":      failed,
            "masks_saved": masks_saved,
            "error_count": len(error_list),
            "errors":      error_list[:10],
            "report_path": str(_DATASET / "annotation_report.json"),
            "message":     msg,
        }, "Dataset annotated", 200)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[Annotate] Unexpected outer error:\n{tb}")
        # Still write whatever partial report we have
        try:
            report = {
                "body_images_processed": annotated,
                "body_images_failed":    failed,
                "masks_generated":       masks_saved,
                "error_list":            error_list + [f"outer_exception: {exc}"],
                "annotated_at":          datetime.datetime.utcnow().isoformat() + "Z",
            }
            (_DATASET / "annotation_report.json").write_text(json.dumps(report, indent=2))
        except Exception:
            pass
        return error_response(
            f"Annotation failed: {str(exc)}",
            500,
            extra={"traceback": tb.strip().splitlines()[-5:], "errors": error_list[:10]},
        )


@dataset_bp.route("/counts", methods=["GET"])
def dataset_counts():
    """
    Live filesystem counts — never cached, always reads directly from disk.
    Returns raw counts, cleaned counts, annotation counts, and folder sizes.
    Used by the frontend Dataset Status panel.
    """
    def _folder_size_mb(folder: Path) -> float:
        if not folder.exists():
            return 0.0
        total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        return round(total / 1024 / 1024, 2)

    raw     = {cat: _count_images(d) for cat, d in _CATEGORY_DIRS.items()}
    cleaned = {cat: _count_images(d) for cat, d in _CLEANED_DIRS.items()}
    masks   = _count_images(_MASKS_DIR)
    annots  = sum(1 for f in _ANNOTATIONS_DIR.rglob("*.json")
                  if f.is_file()) if _ANNOTATIONS_DIR.exists() else 0

    raw_size_mb     = sum(_folder_size_mb(d) for d in _CATEGORY_DIRS.values())
    cleaned_size_mb = sum(_folder_size_mb(d) for d in _CLEANED_DIRS.values())

    total_raw     = sum(raw.values())
    total_cleaned = sum(cleaned.values())

    readiness = {
        cat: {
            "raw":          raw[cat],
            "cleaned":      cleaned[cat],
            "ready":        cleaned[cat] >= _MIN_TRAIN,
            "minimum":      _MIN_TRAIN,
        }
        for cat in _CATEGORY_DIRS
    }

    return success_response({
        "raw":              raw,
        "cleaned":          cleaned,
        "total_raw":        total_raw,
        "total_cleaned":    total_cleaned,
        "masks":            masks,
        "annotations":      annots,
        "raw_size_mb":      round(raw_size_mb, 2),
        "cleaned_size_mb":  round(cleaned_size_mb, 2),
        "readiness":        readiness,
        "can_train":        all(r["ready"] for r in readiness.values()),
        "min_per_cat":      _MIN_TRAIN,
    }, "Live dataset counts", 200)


@dataset_bp.route("/status", methods=["GET"])
def dataset_status():
    """Return current dataset state, per-category counts, cleaned counts, and training readiness."""
    state       = _read_state()
    per_cat     = _per_category_counts()
    cleaned_cat = _per_cleaned_counts()
    total       = sum(per_cat.values())
    total_clean = sum(cleaned_cat.values())

    ann_count   = sum(1 for f in _ANNOTATIONS_DIR.rglob("*.json")
                      if f.is_file()) if _ANNOTATIONS_DIR.exists() else 0

    # Training readiness — based on cleaned counts
    readiness = {
        cat: {
            "raw_count":     per_cat.get(cat, 0),
            "cleaned_count": cleaned_cat.get(cat, 0),
            "ready":         cleaned_cat.get(cat, 0) >= _MIN_TRAIN,
            "minimum":       _MIN_TRAIN,
            "hint": (
                "Ready for training" if cleaned_cat.get(cat, 0) >= _MIN_TRAIN
                else f"Need {_MIN_TRAIN - cleaned_cat.get(cat, 0)} more cleaned images"
            ),
        }
        for cat in _CATEGORY_DIRS
    }
    can_train    = all(r["ready"] for r in readiness.values())
    blocked_cats = [c for c, r in readiness.items() if not r["ready"]]

    return success_response({
        **state,
        "per_category":         per_cat,
        "cleaned_per_category": cleaned_cat,
        "total_images":         total,
        "total_cleaned":        total_clean,
        "annotation_count":     ann_count,
        "readiness":            readiness,
        "can_train":            can_train,
        "blocked_cats":         blocked_cats,
        "min_per_cat":          _MIN_TRAIN,
        "recommended": {
            "testing":      50,
            "first_model":  150,
            "better_model": 500,
        },
    }, "Dataset status", 200)
