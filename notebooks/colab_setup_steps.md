# Google Colab Setup Guide — SareeDrape Studio Training

Use this guide before running `colab_saree_training_10k.ipynb`.  
This is the **safe first run** — 10,000 images per category, 5 epochs, ~30–60 min on Colab GPU.

---

## Step 1 — Open the notebook in Google Colab

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Select `notebooks/colab_saree_training_10k.ipynb` from this project

Or open directly from GitHub:
- **File → Open notebook → GitHub tab**
- Paste: `https://github.com/VaisnaviRajaratnam14/SareeDrap`
- Select `notebooks/colab_saree_training_10k.ipynb`

---

## Step 2 — Enable GPU runtime

```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

> ⚠️ The notebook Cell 1 will raise an error and stop if no GPU is detected.  
> Free Colab gives T4 GPU — sufficient for 10k images × 5 epochs.

---

## Step 3 — Add Kaggle API credentials

The notebook downloads datasets from Kaggle automatically.  
You need to add your Kaggle key as Colab Secrets:

```
Colab left sidebar → 🔑 Secrets (key icon) → + Add new secret
  Name: KAGGLE_USERNAME    Value: your-kaggle-username
  Name: KAGGLE_KEY         Value: your-api-key
```

Get your API key:
1. Go to https://www.kaggle.com/settings
2. Scroll to **API** section
3. Click **Create New Token** → downloads `kaggle.json`
4. Open the file — copy `username` and `key` values

---

## Step 4 — Accept dataset licences on Kaggle (one-time)

Open each link and click **Download** to accept terms:

| Dataset | Link | Used for |
|---|---|---|
| Saree Try-On | https://www.kaggle.com/datasets/chirag2466/saree-tryon-dataset | saree images + body images |
| VITON-HD | https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset | body images + blouse/cloth |
| Human Women | https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women | women full-body with face |
| Indian Saree Patterns | https://www.kaggle.com/datasets/div456/indian-saree-patterns | fabric textures for blouse |

> You only need to accept once per Kaggle account.

---

## Step 5 — Run all cells

```
Runtime → Run all   (Ctrl+F9)
```

Expected runtime breakdown:

| Cell | Task | Time |
|---|---|---|
| Cell 1 | GPU check + install | ~2 min |
| Cell 2 | Config | instant |
| Cell 3 | Download 4 datasets | ~5–10 min |
| Cell 4 | Clean + resize images | ~5–10 min |
| Cell 5 | Pose annotations + masks | ~10–20 min |
| Cell 6 | Model definition | instant |
| Cell 7 | Dataset + DataLoader | instant |
| Cell 8 | Training (5 epochs × 10k) | ~15–30 min |
| Cell 9 | Save config + report | instant |
| Cell 10 | Verify weights | instant |
| Cell 11 | Zip outputs | instant |

**Total: ~40–75 minutes**

---

## Step 6 — Download the trained weights

After Cell 11 completes you will see:
```
✅ saree_weights.zip  →  XXX MB total
```

Download options:

**Option A — Files panel (easiest)**
```
Colab left sidebar → 📁 Files icon → navigate to /content/saree_weights.zip
Right-click → Download
```

**Option B — Direct code (add a new cell at the bottom)**
```python
from google.colab import files
files.download('/content/saree_weights.zip')
```

---

## Step 7 — Deploy weights to local Flask backend

1. **Extract** `saree_weights.zip` on your machine

2. **Copy files** to the project:
   ```
   Windows (cmd/powershell):
   copy saree_tryon_model.pth  sareedrap\backend\dataset\trained_models\
   copy model_config.json      sareedrap\backend\dataset\trained_models\
   copy training_report.json   sareedrap\backend\dataset\trained_models\   (optional)
   ```

   ```
   Mac/Linux:
   cp saree_tryon_model.pth  sareedrap/backend/dataset/trained_models/
   cp model_config.json      sareedrap/backend/dataset/trained_models/
   ```

3. **Restart Flask backend:**
   ```bash
   cd sareedrap/backend
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Mac/Linux
   python app.py
   ```

4. **Confirm ML engine loaded** — startup log should show:
   ```
   ML engine     : available (weights found)
   ```

5. **Test in the UI:**
   ```
   Open http://localhost:3000
   → Upload page → upload body image + saree image + blouse image
   → Customize page → Step 4 → click "Test Inference"
   → Should return: engine = ml_model
   ```

6. **Generate a draping:**
   ```
   Customize page → configure style → Generate Draping
   ```
   The output will now use the trained `SareeTryOnModel` instead of the geometric fallback.

---

## File locations after deployment

```
sareedrap/
└── backend/
    └── dataset/
        └── trained_models/
            ├── saree_tryon_model.pth    ← trained weights (~430 MB)
            ├── model_config.json        ← hyperparameters + metadata
            └── training_report.json     ← epoch log + loss curves
```

---

## Upgrading to full 135k training later

Once the 10k run looks good (loss decreasing, inference working):

1. Open `notebooks/full_saree_training_150k.ipynb` on **Kaggle** (not Colab — needs more disk)
2. Kaggle gives 20 GB disk + P100/T4 GPU free
3. Full run trains on all 4 datasets (~143k total images), 20 epochs
4. Same deployment steps above apply

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA not available` | Runtime → Change runtime type → GPU T4 |
| `403 Forbidden` on dataset download | Accept dataset licence at the Kaggle URL |
| `KeyError: KAGGLE_USERNAME` | Add secrets in Colab left sidebar → 🔑 |
| `0 images after cleaning` | Check dataset download completed — re-run Cell 3 |
| `DataLoader 0 triplets` | At least one category has 0 images — check Cell 3 output |
| Loss stays at `1.0` | Normal for first 1–2 epochs — should drop by epoch 3+ |
| Colab disconnects mid-training | Enable `Tools → Settings → Power saving mode: off` |
| Zip not in Files panel | Run Cell 11 again — or add `files.download('/content/saree_weights.zip')` |
