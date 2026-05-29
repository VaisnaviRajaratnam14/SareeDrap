# AI Virtual Saree Draping — Complete Workflow Guide

**Stack confirmed working:**
- `torch 2.12.0+cpu` ✓
- `torchvision 0.27.0+cpu` ✓
- `mediapipe 0.10.35` ✓
- `opencv-python 4.13.0` ✓
- `kaggle 2.2.0` ✓
- Flask backend on port 5000 ✓
- React frontend on port 3000/5173 ✓

---

## STEP 1 — PyTorch (Already Installed)

```
torch 2.12.0+cpu   ✓
torchvision 0.27.0+cpu   ✓
```

If you ever need to reinstall:
```powershell
cd d:\sareedrap\backend
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

After reinstall, restart Flask:
```powershell
venv\Scripts\python app.py
```

---

## STEP 2 — Kaggle API Setup (REQUIRED before fetching)

### Get your API token

1. Go to **https://www.kaggle.com/settings/api**
2. Click **"Create New Token"**
3. A file `kaggle.json` downloads — contents look like:
   ```json
   {"username":"your_username","key":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
   ```

### Install credentials (choose ONE method)

**Method A — File (recommended):**
```powershell
# Create the .kaggle folder
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"

# Copy the downloaded file
Copy-Item "C:\Users\hp\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

**Method B — Environment variables (add to `backend\.env`):**
```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```
Then restart Flask so `.env` is reloaded.

### Verify credentials work
```powershell
cd d:\sareedrap\backend
venv\Scripts\python -c "
import os, json, pathlib
kj = pathlib.Path.home() / '.kaggle' / 'kaggle.json'
if kj.exists():
    data = json.loads(kj.read_text())
    print('kaggle.json found:', data.get('username','?'))
elif os.environ.get('KAGGLE_USERNAME'):
    print('env vars set:', os.environ['KAGGLE_USERNAME'])
else:
    print('NO CREDENTIALS — follow Method A or B above')
"
```

---

## STEP 3 — Fetch Real Datasets (Frontend)

Open the frontend → **Customize** page → **Model Preparation** panel.

### Recommended real Kaggle datasets

| Category | Slug | Description |
|---|---|---|
| `body_images` | `paramaggarwal/fashion-product-images-small` | Fashion model body shots |
| `body_images` | `validmodel/indian-model-body-images` | Indian body pose images |
| `saree_images` | `bhavyadabas/indian-traditional-cloth-sarees-dataset` | Real saree images |
| `saree_images` | `amitprajapati89/saree-dataset` | Saree fabric photos |
| `blouse_materials` | `gpiosenka/fabric-textures` | Fabric texture close-ups |
| `blouse_materials` | `alexattia/small-scale-benchmark` | Fabric material images |

> **Tip:** You can also search https://www.kaggle.com/datasets for "saree" or "indian fashion"

### Fetch workflow (3 separate fetches)

**Fetch 1 — Body Images:**
1. Slug: `paramaggarwal/fashion-product-images-small`
2. Category: `body_images`
3. Max Images: `150`
4. Click **Fetch Dataset**
5. Wait for: `✓ 150 images imported`

**Fetch 2 — Saree Images:**
1. Slug: `bhavyadabas/indian-traditional-cloth-sarees-dataset`
2. Category: `saree_images`
3. Max Images: `150`
4. Click **Fetch Dataset**
5. Wait for: `✓ 150 images imported`

**Fetch 3 — Blouse Materials:**
1. Slug: `gpiosenka/fabric-textures`
2. Category: `blouse_materials`
3. Max Images: `150`
4. Click **Fetch Dataset**
5. Wait for: `✓ 150 images imported`

### What happens on the backend
- Calls `kaggle datasets download -d <slug> -p <tmp> --unzip`
- Walks extracted files, accepts: `.jpg .jpeg .png .webp .bmp`
- Rejects: size < 1 KB, dimensions < 100×100
- Copies real images to `backend/dataset/raw/<category>/`
- **Zero placeholder/fake images created**

### Expected progress bars after all 3 fetches
```
body_images      ████████████  150  ✓ ready
saree_images     ████████████  150  ✓ ready
blouse_materials ████████████  150  ✓ ready
```

---

## STEP 4 — Validate Counts

After fetching, check `/api/dataset/status`:
```powershell
venv\Scripts\python -c "
import urllib.request, json
with urllib.request.urlopen('http://localhost:5000/api/dataset/status') as r:
    d = json.loads(r.read())['data']
    print('Raw counts:')
    for cat, cnt in d['per_category'].items():
        status = 'READY' if cnt >= 50 else f'NEED {50-cnt} more'
        print(f'  {cat}: {cnt}  ({status})')
    print('Can train (cleaned):', d.get('can_train', False))
"
```

**Minimum required:** 50 per category  
**Recommended:** 150 per category

---

## STEP 5 — Clean Dataset

Click **Clean Dataset** (Step 2 in the UI).

**What it does:**
- Purges any `placeholder_*` files first
- Validates every raw image: extension → size ≥ 1KB → cv2 decode → 100×100 minimum
- Resizes to fit **512×768** (portrait) with `INTER_AREA`
- Normalises brightness using **CLAHE** on LAB L-channel
- Converts all to **PNG**
- Saves clean copies to `backend/dataset/cleaned/<category>/`
- Writes `backend/dataset/cleaning_report.json`

**Expected result:**
```
✓ 148 images saved to cleaned/
  body_images:      148 cleaned ✓
  saree_images:     147 cleaned ✓
  blouse_materials: 149 cleaned ✓
```

Verify on disk:
```powershell
Get-ChildItem backend\dataset\cleaned -Recurse -File | Group-Object DirectoryName | Select Name, Count
```

---

## STEP 6 — Annotate Dataset

Click **Annotate Dataset** (Step 3 in the UI).

**What it does:**
- **MediaPipe Pose** on every cleaned body image:
  - Detects 33 body landmarks (x, y, z, visibility)
  - Saves per-image JSON to `backend/dataset/annotations/<stem>.json`
- **Binary segmentation masks** for saree + blouse images:
  - Thresholds gray channel → 0/255 mask
  - Saves to `backend/dataset/cleaned/masks/<stem>_mask.png`
- Writes `backend/dataset/annotation_report.json`

**Expected result:**
```
✓ 148 body images annotated with pose keypoints
✓ 296 segmentation masks generated
```

Sample annotation file (`annotations/abc123.json`):
```json
{
  "image": "abc123.png",
  "width": 384,
  "height": 512,
  "keypoints": [
    {"x": 0.48, "y": 0.12, "z": -0.003, "visibility": 0.998},
    ...
  ],
  "detected": true
}
```

> If MediaPipe is unavailable, stub JSONs are written so training can proceed.

---

## STEP 7 — Train Model

Click **Train Model** (Step 4 in the UI). Set Epochs (10 recommended for first run).

**Gates before training starts:**
1. ✓ PyTorch installed (`torch 2.12.0+cpu`)
2. ✓ `cleaned/body_images` ≥ 50 images
3. ✓ `cleaned/saree_images` ≥ 50 images
4. ✓ `cleaned/blouse_materials` ≥ 50 images
5. ✓ Annotations exist (set after clicking Annotate)

**Training architecture: `SareeTryOnModel`**
- Convolutional encoder-decoder
- TPS spatial transform grid (5×5)
- MSELoss + Adam + StepLR scheduler
- Runs on CPU (or CUDA if available)

**Live progress shown:**
```
Epoch 3/10 — loss=0.142836  ETA 47s    [████████░░]  34%
```

**Files saved:**
| File | Description |
|---|---|
| `dataset/trained_models/saree_tryon_model.pth` | Best model weights (lowest loss) |
| `dataset/trained_models/model_config.json` | Architecture config |
| `dataset/trained_models/checkpoints/checkpoint_epoch0010.pth` | Epoch checkpoints |

**The `.pth` file contains:**
```python
{
  "epoch": 10,
  "model_state": { ... },   # real weights
  "loss": 0.0843,
  "stub": False,            # guaranteed False for real training
  "trained_at": "...",
  "total_imgs": 444,
  "device": "cpu"
}
```

> **No fake/stub `.pth` files.** If PyTorch is missing, training is blocked with:  
> `"PyTorch not installed. Install torch before real training"`

---

## STEP 8 — Verify ML Engine

Click **Test Inference** or check via API:

```powershell
venv\Scripts\python -c "
import urllib.request, json
with urllib.request.urlopen('http://localhost:5000/api/render/engine-status') as r:
    d = json.loads(r.read())['data']
    print('weights_exists:    ', d['weights_exists'])
    print('model_loaded:      ', d['model_loaded'])
    print('ml_model_available:', d['ml_model_available'])
    print('engine:            ', d['engine'])
    print('is_stub:           ', d.get('is_stub'))
    print('param_count_M:     ', d.get('param_count_m'))
"
```

**After real training, expected output:**
```
weights_exists:     True
model_loaded:       True
ml_model_available: True
engine:             ml_model
is_stub:            False
param_count_M:      2.45
```

**Before training (geometric fallback):**
```
weights_exists:     False
model_loaded:       False
ml_model_available: False
engine:             geometric_fallback
```

---

## STEP 9 — Test Inference

Click **Test Inference** (Step 5 in the UI) or:

```powershell
venv\Scripts\python -c "
import urllib.request, json
req = urllib.request.Request('http://localhost:5000/api/render/test-inference',
    data=b'{}', headers={'Content-Type':'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=30) as r:
    d = json.loads(r.read())['data']
    print('passed:    ', d['passed'])
    print('latency_ms:', d.get('latency_ms'))
    print('engine:    ', d.get('engine'))
    if not d['passed']:
        print('error:     ', d.get('error'))
"
```

**Passed:**
```
passed:     True
latency_ms: 312.4
engine:     ml_model
```

**Failed (no weights yet):**
```
passed:     False
engine:     geometric_fallback
error:      Weights not found
```

---

## STEP 10 — Generate Draping

**Upload first:**
1. Go to **Upload** page
2. Upload a body photo (person standing)
3. Upload a saree image
4. Upload a blouse/fabric image

**Then generate:**
- Go to **Customize** page
- Choose draping style (Nivi / Bridal / Hanging / Gujarati)
- Choose neck type and sleeve length
- Click **Generate Draping** (Step 6)

**Engine selection logic:**
```
IF ml_model_available == True AND model_loaded == True AND is_stub == False:
    use SareeTryOnModel (ML inference)
ELSE:
    use geometric_fallback (OpenCV warp + blend)
```

**Response:**
```json
{
  "success": true,
  "engine_used": "ml_model",
  "ml_model_available": true,
  "output_path": "outputs/drape_abc123.png",
  "fallback_reason": ""
}
```

---

## STEP 11 — Final Validation

Run this complete validation script:

```powershell
cd d:\sareedrap\backend
venv\Scripts\python -c "
import urllib.request, json, pathlib, os
BASE = 'http://localhost:5000/api'
ROOT = pathlib.Path('.')
ok   = []
fail = []

def chk(label, passed, detail=''):
    if passed: ok.append(label);  print(f'PASS  {label}' + (f'  [{detail}]' if detail else ''))
    else:      fail.append(label); print(f'FAIL  {label}' + (f'  [{detail}]' if detail else ''))

def get(path):
    with urllib.request.urlopen(f'{BASE}{path}', timeout=10) as r:
        return json.loads(r.read()).get('data', {})

# 1. No placeholder files
fake = list((ROOT/'dataset'/'raw').rglob('placeholder_*')) if (ROOT/'dataset'/'raw').exists() else []
chk('No placeholder images', len(fake)==0, f'{len(fake)} found')

# 2. Slug required
import urllib.error
req = urllib.request.Request(f'{BASE}/dataset/kaggle-download',
    data=json.dumps({'category':'body_images'}).encode(),
    headers={'Content-Type':'application/json'}, method='POST')
try:
    urllib.request.urlopen(req, timeout=8)
    chk('Slug required gate', False, 'no 400 returned')
except urllib.error.HTTPError as e:
    chk('Slug required gate', e.code==400, f'HTTP {e.code}')

# 3. Folder structure
for rel in ['dataset/raw/body_images','dataset/raw/saree_images','dataset/raw/blouse_materials',
            'dataset/cleaned/body_images','dataset/cleaned/saree_images','dataset/cleaned/blouse_materials',
            'dataset/cleaned/masks','dataset/annotations','dataset/trained_models']:
    chk(f'Folder {rel}', (ROOT/rel).exists())

# 4. Dataset status endpoint
ds = get('/dataset/status')
chk('Dataset status OK', 'per_category' in ds)
chk('cleaned_per_category present', 'cleaned_per_category' in ds)

# 5. Engine status
es = get('/render/engine-status')
chk('Engine status OK', 'weights_exists' in es, es.get('engine',''))

# 6. Test inference responds
req2 = urllib.request.Request(f'{BASE}/render/test-inference',
    data=b'{}', headers={'Content-Type':'application/json'}, method='POST')
with urllib.request.urlopen(req2, timeout=30) as r:
    ti = json.loads(r.read()).get('data',{})
chk('Test inference responds', 'passed' in ti, str(ti.get('engine','')))

# 7. Train blocks without cleaned data (correct behavior)
req3 = urllib.request.Request(f'{BASE}/train/start',
    data=json.dumps({'epochs':1}).encode(),
    headers={'Content-Type':'application/json'}, method='POST')
try:
    urllib.request.urlopen(req3, timeout=8)
    chk('Train gate check', True, 'started (dataset ready)')
except urllib.error.HTTPError as e:
    d3 = json.loads(e.read())
    chk('Train gate check', e.code in (400,500), f'HTTP {e.code}: {d3.get(\"error\",\"\")[:50]}')

# Summary
print()
print('='*50)
print(f'PASSED: {len(ok)}/{len(ok)+len(fail)}')
if fail:
    print('FAILED:', fail)
else:
    print('ALL CHECKS PASSED')
print('='*50)
"
```

---

## Quick Reference — All API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend health |
| `POST` | `/api/dataset/kaggle-download` | Fetch real images from Kaggle |
| `POST` | `/api/dataset/clean` | Clean raw → cleaned/ |
| `POST` | `/api/dataset/annotate` | Pose keypoints + masks |
| `GET` | `/api/dataset/status` | Counts, readiness, state |
| `POST` | `/api/train/start` | Start real PyTorch training |
| `GET` | `/api/train/status` | Epoch, loss, ETA, progress |
| `GET` | `/api/render/engine-status` | ML engine verification |
| `POST` | `/api/render/test-inference` | Lightweight inference test |
| `POST` | `/api/render/generate` | Full draping generation |

## Directory Structure

```
backend/
  dataset/
    raw/
      body_images/          ← Kaggle downloads land here
      saree_images/
      blouse_materials/
    cleaned/
      body_images/          ← Validated, resized, CLAHE, PNG
      saree_images/
      blouse_materials/
      masks/                ← Segmentation masks
    annotations/            ← MediaPipe keypoint JSONs
    trained_models/
      saree_tryon_model.pth ← Real model weights
      model_config.json
      checkpoints/
    cleaning_report.json
    annotation_report.json
```

## Current System State

| Component | Status |
|---|---|
| Flask backend | ✓ Running on port 5000 |
| PyTorch | ✓ `2.12.0+cpu` installed |
| MediaPipe | ✓ `0.10.35` installed |
| OpenCV | ✓ `4.13.0` installed |
| Kaggle package | ✓ `2.2.0` installed |
| Kaggle credentials | ⚠ **Not configured** — follow Step 2 above |
| Raw dataset | ⚠ Empty — fetch after credentials set |
| Cleaned dataset | ⚠ Empty — run after fetch |
| Annotations | ⚠ Empty — run after clean |
| Trained model | ⚠ Not trained — run after annotate |
| ML engine | → Geometric fallback (until trained) |
