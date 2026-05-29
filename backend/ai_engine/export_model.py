"""
Export Model Script
====================
Run this after training on Kaggle to package the model weights
into the format expected by ml_saree_inference.py.

Usage (from backend/):
    python ai_engine/export_model.py --checkpoint checkpoints/best_model.pth

Output:
    dataset/trained_models/saree_tryon_model.pth
    dataset/trained_models/model_config.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add backend root to path so imports work when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def export(checkpoint_path: str, output_dir: str | None = None) -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "dataset" / "trained_models"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    # Support both formats:
    #   1. Direct state_dict  (keys are layer names)
    #   2. Wrapped checkpoint (keys: model_state, epoch, history, …)
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        meta       = {k: v for k, v in ckpt.items() if k != "model_state"}
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta       = {}
    else:
        # Assume raw state dict
        state_dict = ckpt
        meta       = {}

    # ── Validate model loads correctly ────────────────────────────────────────
    print("Validating model architecture…")
    from ai_engine.ml_models import SareeTryOnModel

    model = SareeTryOnModel(img_h=256, img_w=192, grid_size=5, base_ch=64)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  ⚠  Missing keys  ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  ⚠  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        print("  ✓ State dict matches model exactly")

    # ── Save model weights ────────────────────────────────────────────────────
    out_weights = output_dir / "saree_tryon_model.pth"
    torch.save({"model_state": state_dict, **meta}, str(out_weights))
    size_mb = out_weights.stat().st_size / 1e6
    print(f"  ✓ Saved weights: {out_weights}  ({size_mb:.1f} MB)")

    # ── Save config ───────────────────────────────────────────────────────────
    config = {
        "img_h":   256,
        "img_w":   192,
        "grid_size": 5,
        "base_ch":  64,
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std":  [0.5, 0.5, 0.5],
        "fabric_types":   SareeTryOnModel.FABRIC_TYPES,
        "draping_styles": SareeTryOnModel.DRAPING_STYLES,
        "source_checkpoint": str(ckpt_path.name),
    }
    # Merge any training metadata saved in the checkpoint
    if "history" in meta:
        final_ep = len(meta["history"].get("tom_loss", []))
        config["trained_epochs"]    = final_ep
        config["final_val_loss"]    = meta["history"].get("val_loss", [None])[-1]

    out_cfg = output_dir / "model_config.json"
    with open(out_cfg, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved config:  {out_cfg}")

    # ── Quick smoke test ──────────────────────────────────────────────────────
    print("Running smoke test (CPU forward pass)…")
    model.eval()
    with torch.no_grad():
        dummy = {
            "person":       torch.randn(1, 3,  256, 192),
            "saree":        torch.randn(1, 3,  256, 192),
            "blouse":       torch.randn(1, 3,  256, 192),
            "pose_heatmap": torch.randn(1, 18, 256, 192),
            "seg_mask":     torch.randn(1, 1,  256, 192),
            "fabric_idx":   torch.tensor([0]),
            "style_idx":    torch.tensor([0]),
        }
        out = model(**dummy)
    print(f"  ✓ rendered:     {out['rendered'].shape}")
    print(f"  ✓ warped_saree: {out['warped_saree'].shape}")
    print(f"  ✓ alpha_mask:   {out['alpha_mask'].shape}")
    print("\n✅ Export complete. Place the files in backend/dataset/trained_models/")
    print(f"   {out_weights}")
    print(f"   {out_cfg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained SareeTryOnModel")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pth file (from Kaggle training)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: dataset/trained_models/)")
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir)
