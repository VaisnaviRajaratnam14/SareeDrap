"""
Training routes: start training and check status.
POST /api/train/start
GET  /api/train/status
"""

import json
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone
from flask import Blueprint, request
from utils.helpers import success_response, error_response

train_bp = Blueprint("train", __name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE          = Path(__file__).parent.parent
_WEIGHTS_PATH  = _BASE / "dataset" / "trained_models" / "saree_tryon_model.pth"
_CONFIG_PATH   = _BASE / "dataset" / "trained_models" / "model_config.json"
_STATUS_FILE   = _BASE / "dataset" / "trained_models" / ".train_status.json"
_CKPT_DIR      = _BASE / "dataset" / "trained_models" / "checkpoints"

# Training reads from cleaned/ — not raw/
_CATEGORY_DIRS = {
    "body_images":      _BASE / "dataset" / "cleaned" / "body_images",
    "saree_images":     _BASE / "dataset" / "cleaned" / "saree_images",
    "blouse_materials": _BASE / "dataset" / "cleaned" / "blouse_materials",
}

_MIN_IMAGES       = 50    # hard block per category
_MAX_PROTO_IMAGES = 500   # prototype mode cap per category
_KAGGLE_THRESHOLD = 5000  # total images above which local training is blocked
_IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Model spatial dimensions (must match SareeTryOnModel defaults)
_IMG_H = 256
_IMG_W = 192

# ── Training state ─────────────────────────────────────────────────────────────
_train_lock   = threading.Lock()
_train_thread = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*")
               if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)


def _read_status() -> dict:
    if _STATUS_FILE.exists():
        try:
            return json.loads(_STATUS_FILE.read_text())
        except Exception:
            pass
    model_exists = (
        _WEIGHTS_PATH.exists() and _WEIGHTS_PATH.stat().st_size > 1024
    ) if _WEIGHTS_PATH.exists() else False
    return {
        "status":       "completed" if model_exists else "idle",
        "model_exists": model_exists,
        "progress":     100 if model_exists else 0,
        "epoch":        0,
        "total_epochs": 0,
        "loss":         None,
        "eta_seconds":  None,
        "message":      "Weights found" if model_exists else "Not trained yet",
        "started_at":   None,
        "finished_at":  None,
    }


def _write_status(patch: dict):
    current = _read_status()
    current.update(patch)
    _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_FILE.write_text(json.dumps(current, indent=2))


# ── Dataset helper ─────────────────────────────────────────────────────────────

def _build_dataset(device, max_per_folder=None):
    """
    Load and preprocess cleaned images into tensors.

    Parameters
    ----------
    device         : torch.device
    max_per_folder : int | None — cap images per category (prototype mode)

    Returns three lists:
      body_tensors   – list of (3, H, W) tensors
      saree_tensors  – list of (3, H, W) tensors
      blouse_tensors – list of (3, H, W) tensors
    """
    import torch
    import cv2
    import numpy as np

    def _load_folder(folder: Path):
        tensors = []
        paths   = sorted(
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        if max_per_folder:
            paths = paths[:max_per_folder]
        for fpath in paths:
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = cv2.resize(img, (_IMG_W, _IMG_H), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr = img.astype(np.float32) / 127.5 - 1.0   # → [-1, 1]
            t   = torch.from_numpy(arr.transpose(2, 0, 1)).to(device)  # (3,H,W)
            tensors.append(t)
        return tensors

    body   = _load_folder(_CATEGORY_DIRS["body_images"])
    saree  = _load_folder(_CATEGORY_DIRS["saree_images"])
    blouse = _load_folder(_CATEGORY_DIRS["blouse_materials"])
    return body, saree, blouse


# ── Real training worker ───────────────────────────────────────────────────────

def _run_training(config: dict):
    """
    Background training worker — real PyTorch only, no stubs.

    Architecture: SareeTryOnModel (VITON-HD style)
    Loss:  pixel reconstruction (MSE) on rendered output vs body image
    Saves: best model weights + config to trained_models/

    Modes
    -----
    prototype  — local CPU, ≤500 images per category, marks output as prototype
    kaggle_full — must not run here (blocked at start_training route level)
    """
    import torch

    total_epochs    = config.get("epochs", 10)
    draping_style   = config.get("draping_style", "nivi")
    training_mode   = config.get("training_mode", "prototype")   # prototype | kaggle_full
    is_prototype    = training_mode == "prototype"
    max_imgs        = _MAX_PROTO_IMAGES if is_prototype else None  # None = no cap
    started_at      = _now()

    print(f"[Train] torch version  : {torch.__version__}")
    print(f"[Train] cuda available : {torch.cuda.is_available()}")
    print(f"[Train] mode           : {training_mode}  (is_prototype={is_prototype})")
    print(f"[Train] Starting — epochs={total_epochs}  style={draping_style}  max_imgs_per_cat={max_imgs}")

    _write_status({
        "status":        "running",
        "model_exists":  False,
        "progress":      0,
        "epoch":         0,
        "total_epochs":  total_epochs,
        "loss":          None,
        "eta_seconds":   None,
        "training_mode": training_mode,
        "is_prototype":  is_prototype,
        "message":       f"Initialising — {training_mode} mode, loading model and images…",
        "started_at":    started_at,
        "finished_at":   None,
    })

    try:
        from ai_engine.ml_models import SareeTryOnModel

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Train] Device: {device}")
        _write_status({"message": f"Loading SareeTryOnModel on {device}…", "progress": 3})

        model     = SareeTryOnModel(img_h=_IMG_H, img_w=_IMG_W, grid_size=5, base_ch=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_epochs // 3), gamma=0.5
        )

        _CKPT_DIR.mkdir(parents=True, exist_ok=True)
        _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        # ── Load images ────────────────────────────────────────────────────────
        cap_msg = f" (capped at {max_imgs}/category)" if max_imgs else ""
        _write_status({"message": f"Loading cleaned images{cap_msg}…", "progress": 5})
        body_imgs, saree_imgs, blouse_imgs = _build_dataset(device, max_per_folder=max_imgs)

        n_body   = len(body_imgs)
        n_saree  = len(saree_imgs)
        n_blouse = len(blouse_imgs)
        print(f"[Train] Images loaded — body={n_body}  saree={n_saree}  blouse={n_blouse}")

        if n_body == 0 or n_saree == 0 or n_blouse == 0:
            raise ValueError(
                f"Empty category after loading: body={n_body}, saree={n_saree}, blouse={n_blouse}. "
                "Run Fetch → Clean first."
            )

        total_imgs = n_body + n_saree + n_blouse
        style_idx  = torch.tensor(
            [SareeTryOnModel.style_index(draping_style)], device=device
        )
        _write_status({
            "message": f"Training starts — {total_imgs} images on {device}",
            "progress": 8,
        })

        # ── Training loop ──────────────────────────────────────────────────────
        epoch_times = []
        best_loss   = float("inf")
        batch_size  = 2    # keep memory small for CPU

        model.train()
        for epoch in range(1, total_epochs + 1):
            t_ep = time.time()
            epoch_loss_sum = 0.0
            steps = 0

            # Iterate body images, cycling saree/blouse if shorter
            for i in range(0, n_body, batch_size):
                body_batch   = torch.stack(body_imgs[i:i + batch_size])          # (B,3,H,W)
                B            = body_batch.size(0)
                saree_batch  = torch.stack([saree_imgs[j % n_saree]
                                            for j in range(i, i + B)])
                blouse_batch = torch.stack([blouse_imgs[j % n_blouse]
                                            for j in range(i, i + B)])

                # Synthetic pose heatmap and seg mask (training signal)
                pose_heatmap = torch.zeros(B, 18, _IMG_H, _IMG_W, device=device)
                seg_mask     = torch.ones( B,  1, _IMG_H, _IMG_W, device=device)
                s_idx        = style_idx.expand(B)

                optimizer.zero_grad()
                outputs = model(
                    person       = body_batch,
                    saree        = saree_batch,
                    blouse       = blouse_batch,
                    pose_heatmap = pose_heatmap,
                    seg_mask     = seg_mask,
                    style_idx    = s_idx,
                )
                # Reconstruction loss: rendered image should resemble body image
                loss = criterion(outputs["rendered"], body_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss_sum += float(loss.item())
                steps += 1

            scheduler.step()

            epoch_time  = time.time() - t_ep
            epoch_times.append(epoch_time)
            loss_val    = round(epoch_loss_sum / max(steps, 1), 6)
            avg_time    = sum(epoch_times) / len(epoch_times)
            remaining   = int(avg_time * (total_epochs - epoch))
            progress    = 8 + int(87 * epoch / total_epochs)

            print(f"[Train] Epoch {epoch}/{total_epochs}  loss={loss_val:.6f}  "
                  f"ETA={remaining}s  steps={steps}")

            # Save checkpoint every 5 epochs or last epoch
            if epoch % 5 == 0 or epoch == total_epochs:
                ckpt_path = _CKPT_DIR / f"checkpoint_epoch{epoch:04d}.pth"
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "loss":         loss_val,
                    "stub":         False,
                }, str(ckpt_path))
                print(f"[Train] Checkpoint saved: {ckpt_path.name}")

            # Save best model
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "loss":         loss_val,
                    "stub":         False,
                    "is_prototype": is_prototype,
                    "trained_at":   started_at,
                }, str(_WEIGHTS_PATH))

            _write_status({
                "progress":     progress,
                "epoch":        epoch,
                "total_epochs": total_epochs,
                "loss":         loss_val,
                "best_loss":    round(best_loss, 6),
                "eta_seconds":  remaining,
                "message":      f"Epoch {epoch}/{total_epochs} — loss={loss_val:.6f}  ETA {remaining}s",
            })

        # ── Final model save ───────────────────────────────────────────────────
        torch.save({
            "epoch":        total_epochs,
            "model_state":  model.state_dict(),
            "loss":         round(best_loss, 6),
            "stub":         False,
            "is_prototype": is_prototype,
            "trained_at":   started_at,
            "total_imgs":   total_imgs,
            "device":       str(device),
            "img_h":        _IMG_H,
            "img_w":        _IMG_W,
        }, str(_WEIGHTS_PATH))

        proto_note = " [PROTOTYPE — pipeline test only]" if is_prototype else ""
        cfg = {
            "img_h":         _IMG_H,
            "img_w":         _IMG_W,
            "grid_size":     5,
            "base_ch":       64,
            "epochs":        total_epochs,
            "device":        str(device),
            "stub":          False,
            "is_prototype":  is_prototype,
            "training_mode": training_mode,
            "trained_at":    started_at,
            "total_imgs":    total_imgs,
            "note":          proto_note.strip() if is_prototype else "Full local training",
        }
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

        size_kb  = _WEIGHTS_PATH.stat().st_size / 1024
        done_msg = (f"Training complete{proto_note} — {total_epochs} epochs, "
                    f"best_loss={best_loss:.6f}, device={device}, "
                    f"weights={size_kb:.0f} KB")
        print(f"[Train] ✅ {done_msg}")

        _write_status({
            "status":        "completed",
            "model_exists":  True,
            "is_prototype":  is_prototype,
            "training_mode": training_mode,
            "progress":      100,
            "epoch":         total_epochs,
            "loss":          round(best_loss, 6),
            "eta_seconds":   0,
            "message":       done_msg,
            "finished_at":   _now(),
            "weights_kb":    round(size_kb, 1),
        })

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[Train] ❌ Training failed:\n{tb}")
        _write_status({
            "status":       "failed",
            "model_exists": False,
            "progress":     0,
            "message":      f"Training failed: {exc}",
            "error":        str(exc),
            "traceback":    tb.strip().splitlines()[-5:],
            "finished_at":  _now(),
        })


# ── Routes ─────────────────────────────────────────────────────────────────────

@train_bp.route("/start", methods=["POST"])
def start_training():
    """
    Start model training in a background thread.
    Body (JSON, all optional):
      draping_style  – hint for training config
      epochs         – number of training epochs (default 10)
    """
    global _train_thread

    with _train_lock:
        status = _read_status()
        if status.get("status") == "running":
            return success_response(status, "Training already in progress", 200)

    # ── Check PyTorch is available ────────────────────────────────────────────
    try:
        import torch as _t
        print(f"[Train] torch version : {_t.__version__}")
        print(f"[Train] cuda available: {_t.cuda.is_available()}")
    except ImportError as ie:
        return error_response(
            f"PyTorch not installed in this Python environment: {ie}. "
            "Run: pip install torch torchvision",
            500,
            extra={"code": "PYTORCH_MISSING"},
        )

    body          = request.get_json(silent=True) or {}
    training_mode = body.get("training_mode", "prototype")   # prototype | kaggle_full

    # ── Block kaggle_full mode on local machine ────────────────────────────────
    if training_mode == "kaggle_full":
        return error_response(
            "Full 150k training must be done on Kaggle GPU, not local CPU. "
            "Open notebooks/full_saree_training_150k.ipynb in Kaggle, run it, "
            "then download saree_tryon_model.pth and place it in "
            "backend/dataset/trained_models/",
            400,
            extra={
                "code":         "KAGGLE_TRAINING_REQUIRED",
                "notebook":     "notebooks/full_saree_training_150k.ipynb",
                "weights_dest": "backend/dataset/trained_models/saree_tryon_model.pth",
            },
        )

    # ── Dataset readiness gate (cleaned images) ───────────────────────────────
    per_cat   = {cat: _count_images(folder) for cat, folder in _CATEGORY_DIRS.items()}
    total_all = sum(per_cat.values())

    # Block local training if dataset is too large (>5000 total)
    if total_all > _KAGGLE_THRESHOLD:
        return error_response(
            f"Dataset has {total_all} images — 150k training must be done on Kaggle GPU, "
            "not local CPU. Use the Kaggle notebook for full training.",
            400,
            extra={
                "code":          "DATASET_TOO_LARGE_FOR_LOCAL",
                "total_images":  total_all,
                "threshold":     _KAGGLE_THRESHOLD,
                "notebook":      "notebooks/full_saree_training_150k.ipynb",
                "per_category":  per_cat,
            },
        )

    blocked = {cat: cnt for cat, cnt in per_cat.items() if cnt < _MIN_IMAGES}
    if blocked:
        details = {cat: {"count": cnt, "need": _MIN_IMAGES - cnt}
                   for cat, cnt in blocked.items()}
        return error_response(
            "Insufficient cleaned dataset — fetch, clean, and annotate before training",
            400,
            extra={
                "code":         "INSUFFICIENT_DATASET",
                "per_category": per_cat,
                "blocked":      details,
                "min_per_cat":  _MIN_IMAGES,
                "hint":         "Run: Fetch Dataset → Clean Dataset → Annotate → Train",
            },
        )

    config = {
        "draping_style": body.get("draping_style", "nivi"),
        "epochs":        max(1, int(body.get("epochs") or 10)),
        "training_mode": training_mode,
    }

    with _train_lock:
        _train_thread = threading.Thread(
            target=_run_training, args=(config,), daemon=True
        )
        _train_thread.start()

    is_proto = training_mode == "prototype"
    return success_response({
        "status":        "running",
        "training_mode": training_mode,
        "is_prototype":  is_proto,
        "message":       (
            f"Prototype training started (≤{_MAX_PROTO_IMAGES} imgs/category). "
            "Poll /api/train/status for updates."
            if is_proto else
            "Training started. Poll /api/train/status for updates."
        ),
        "config":       config,
        "per_category": per_cat,
    }, "Training started", 202)


@train_bp.route("/status", methods=["GET"])
def training_status():
    """Return current training status with epoch, loss, ETA, and per-category counts."""
    status  = _read_status()
    per_cat = {cat: _count_images(folder) for cat, folder in _CATEGORY_DIRS.items()}
    blocked = {cat: cnt for cat, cnt in per_cat.items() if cnt < _MIN_IMAGES}

    status["model_exists"]  = (
        _WEIGHTS_PATH.exists() and _WEIGHTS_PATH.stat().st_size > 1024
    ) if _WEIGHTS_PATH.exists() else False
    status["per_category"]  = per_cat
    status["can_train"]     = len(blocked) == 0
    status["min_per_cat"]   = _MIN_IMAGES
    status["blocked_cats"]  = list(blocked.keys())

    # Check for latest checkpoint
    if _CKPT_DIR.exists():
        ckpts = sorted(_CKPT_DIR.glob("checkpoint_epoch*.pth"))
        status["latest_checkpoint"] = ckpts[-1].name if ckpts else None
        status["checkpoint_count"]  = len(ckpts)
    else:
        status["latest_checkpoint"] = None
        status["checkpoint_count"]  = 0

    return success_response(status, "Training status", 200)
