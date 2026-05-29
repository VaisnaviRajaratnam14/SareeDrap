# SareeDrape Studio — AI Virtual Saree Draping System

A full-stack AI-powered virtual saree draping and blouse customization platform built with **React**, **Flask**, **MongoDB**, **MediaPipe**, and **OpenCV**.

---

## Project Structure

```
sareedrap/
├── frontend/               React + Tailwind CSS UI
│   ├── src/
│   │   ├── pages/          HomePage, UploadPage, CustomizePage, PreviewPage, HistoryPage, AdminDashboard
│   │   ├── components/     Navbar, UploadZone, StepIndicator, LoadingSpinner
│   │   ├── context/        AuthContext, DrapingContext
│   │   └── services/       api.js (Axios)
│   ├── public/
│   ├── tailwind.config.js
│   └── package.json
│
├── backend/                Flask REST API
│   ├── app.py              Application factory
│   ├── seed.py             Database seed script
│   ├── routes/
│   │   ├── auth_routes.py      Register/Login/Profile
│   │   ├── upload_routes.py    Body/Saree/Blouse upload
│   │   ├── pose_routes.py      Pose detection trigger
│   │   ├── saree_routes.py     Saree feature extraction + templates
│   │   ├── blouse_routes.py    Blouse customization + templates
│   │   ├── render_routes.py    Full pipeline render + download
│   │   ├── admin_routes.py     Admin CRUD
│   │   └── static_routes.py    Static file serving
│   ├── ai_engine/
│   │   ├── pose_detection.py   MediaPipe 33-landmark pose detection
│   │   ├── segmentation.py     rembg / GrabCut body segmentation
│   │   ├── saree_draping.py    Pleat/pallu generation, 4 draping styles
│   │   ├── blouse_fitting.py   Neck/sleeve fitting with colour tinting
│   │   └── rendering.py        Full composite pipeline
│   ├── models/
│   │   └── schemas.py          MongoDB document schemas
│   ├── utils/
│   │   └── helpers.py          JWT, file utils, response helpers
│   └── requirements.txt
│
├── dataset/
│   ├── body_samples/       Sample body images for testing
│   ├── saree_samples/      Sample saree textures
│   └── blouse_samples/     Sample blouse materials
│
├── uploads/                Temporary uploaded images (auto-deleted)
├── outputs/                Generated draping outputs
└── models/                 Pretrained model files (optional)
```

---

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **MongoDB** (local or Atlas)

---

## Installation & Setup

### 1. Clone / Navigate to project

```bash
cd sareedrap
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Edit .env with your MongoDB URI and SECRET_KEY

# Seed the database
python seed.py

# Start Flask server
python app.py
```

Backend runs on: **http://localhost:5000**

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start React dev server
npm start
```

Frontend runs on: **http://localhost:3000**

---

## MongoDB Atlas Setup (Cloud — Recommended)

> Skip this if you already have a local MongoDB running.

### Step-by-step

1. **Create a free cluster**  
   Go to [https://cloud.mongodb.com](https://cloud.mongodb.com) → **Create a free M0 cluster**

2. **Create a database user**  
   *Database Access* → *Add New Database User*  
   - Username: `saree_user`  
   - Password: (copy it)  
   - Role: **Read and write to any database**

3. **Whitelist your IP**  
   *Network Access* → *Add IP Address* → **Allow access from anywhere** (`0.0.0.0/0`) for development

4. **Copy the connection string**  
   *Clusters* → *Connect* → *Drivers* → copy the URI that looks like:  
   ```
   mongodb+srv://saree_user:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```

5. **Configure `.env`**  
   Copy `backend/.env.example` to `backend/.env` and set:
   ```env
   MONGO_URI=mongodb+srv://saree_user:<password>@cluster0.xxxxx.mongodb.net/saree_draping_db?retryWrites=true&w=majority
   SECRET_KEY=your-random-secret-key
   ```

6. **Seed demo accounts**
   ```bash
   cd backend
   venv\Scripts\activate       # Windows
   python seed.py
   ```

7. **Verify connection**  
   Start the backend — you should see:
   ```
   ════════════════════════════════════════════════════
     SareeDrape Studio — AI Saree Draping System
   ════════════════════════════════════════════════════
     Backend       : starting on port 5000
     Database      : Connecting to MongoDB...
     URI type      : MongoDB Atlas (cloud)
     Database      : ✓ Connected  (db=saree_draping_db)
     ML engine     : fallback (geometric fallback — no weights)
   ────────────────────────────────────────────────────
     Auth/history  : enabled
     AI processing : enabled (always)
     Health check  : GET /api/health
   ════════════════════════════════════════════════════
   ```

### Graceful fallback behaviour

| MongoDB status | Auth / Login | AI Processing | File Uploads |
|---|---|---|---|
| ✓ Connected | ✓ Works | ✓ Works | ✓ Works |
| ✗ Unavailable | ✗ Returns 503 | ✓ Works | ✓ Works |

The draping engine (`/api/render/generate`, `/api/upload/*`, `/api/pose/*`) always works regardless of database status.

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/api/health` | Backend + DB + ML engine status |
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login, receive JWT |
| GET  | `/api/auth/profile` | Get current user |
| POST | `/api/upload/body` | Upload body image |
| POST | `/api/upload/saree` | Upload saree image |
| POST | `/api/upload/blouse` | Upload blouse material |
| POST | `/api/pose/detect` | Run MediaPipe pose detection |
| POST | `/api/saree/process` | Extract saree features |
| GET  | `/api/saree/templates` | List saree templates |
| POST | `/api/blouse/customize` | Fit blouse on body |
| GET  | `/api/blouse/templates` | List blouse templates |
| POST | `/api/render/generate` | Full draping pipeline (ML or fallback) |
| GET  | `/api/render/engine-status` | Which engine will be used |
| GET  | `/api/render/download/:id` | Download output image |
| DELETE | `/api/render/cleanup/:id` | Delete temp files |
| GET  | `/api/render/history` | User's draping history |
| POST | `/api/process/run` | Alternative ML pipeline route |
| GET  | `/api/process/engine-status` | ML engine status |
| GET  | `/api/admin/stats` | Dashboard statistics |
| GET  | `/api/admin/users` | List all users |
| GET  | `/api/admin/outputs` | List all outputs |

---

## MongoDB Collections

| Collection | Purpose |
|---|---|
| `users` | User accounts with hashed passwords and roles |
| `sarees` | Saree templates and metadata |
| `blouses` | Blouse templates and customization data |
| `generated_outputs` | Draping session records with pose data |

---

## AI Pipeline Flow

```
Body Image Upload
      ↓
MediaPipe Pose Detection (33 keypoints)
      ↓
rembg / GrabCut Body Segmentation
      ↓
Saree Feature Extraction
  (dominant colour, border, texture patch)
      ↓
Draping Engine (Nivi / Bridal / Hanging / Gujarati)
  ├── Pleat generation (geometric strips + perspective)
  ├── Pallu overlay (rotation transform)
  └── Fabric simulation (silk sheen / cotton noise / chiffon blur)
      ↓
Blouse Fitting
  ├── Colour tinting
  ├── Neck cutout (polygon mask)
  └── Sleeve overlay (arm vector interpolation)
      ↓
Final Composite + Post-Processing
  (CLAHE → unsharp mask → saturation boost)
      ↓
Save Output → Return path → Download
```

---

## Draping Styles

| Style | Description | Region |
|---|---|---|
| **Nivi** | Classic front pleats, pallu over left shoulder | South/North India |
| **Bridal** | Dense pleats, wide pallu spread | Pan India |
| **Hanging** | Loose hip drape, flowing pleats | East India |
| **Gujarati** | Front pallu over right shoulder | West India |

---

## Demo Credentials

| Role | Email | Password |
|---|---|---|
| Admin | admin@saree.com | admin123 |
| User | demo@saree.com | demo123 |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Tailwind CSS, React Router v6, Axios |
| Backend | Flask 3, Flask-CORS, Flask-PyMongo |
| Database | MongoDB (PyMongo) |
| AI/CV | MediaPipe, OpenCV, rembg (U2-Net), scikit-image |
| Auth | JWT (PyJWT), bcrypt |
| 3D/Viz | Three.js (@react-three/fiber) — available for extension |

---

## Environment Variables

### Backend `.env`
```
SECRET_KEY=your-secret-key
MONGO_URI=mongodb://localhost:27017/saree_draping_db
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
ALLOWED_ORIGINS=http://localhost:3000
FLASK_DEBUG=true
JWT_EXPIRY_HOURS=24
```

### Frontend `.env`
```
REACT_APP_API_URL=http://localhost:5000/api
```

---

## ML Training — Two Modes

The system supports two training modes selectable from the UI (**Customize → Step 4 → Training Mode**).

| Mode | Images | Hardware | Purpose | Output flag |
|---|---|---|---|---|
| **Prototype Local** | 150–500/category | CPU | Pipeline verification | `is_prototype: true` |
| **Full Kaggle GPU** | Up to 150k/category | GPU T4/P100 | Production model | `is_prototype: false` |

> ⚠️ **Do NOT train the 150k dataset locally.** The backend will block it if total cleaned images exceed 5 000.  
> Train on Kaggle, download the weights, and place them in `backend/dataset/trained_models/`.

---

### Mode 1 — Prototype Local Training (150–500 images)

**Purpose:** Verifies the entire pipeline end-to-end on your machine before committing GPU hours.

#### Steps

```
UI → Customize Page → Step 1: Fetch Dataset (set max 150–500 per category)
                    → Step 2: Clean Dataset
                    → Step 3: Annotate Dataset
                    → Step 4: Training Mode = "Prototype Local" → Train Prototype
```

- Images are capped at **500 per category** automatically in prototype mode.
- Output is marked `is_prototype: true` in the saved `.pth`.
- Engine status will show **ML Model Active** — valid for inference testing.
- Training takes ~5–15 minutes per epoch on CPU depending on hardware.

---

### Mode 2 — Full Kaggle GPU Training (150k images)

**Purpose:** Production-quality model trained on the full dataset.

#### Steps

1. **Select mode** in the UI → Step 4 → **Full Kaggle GPU**  
   The local Train button is disabled; instructions are shown in the UI.

2. **Open the notebook** on Kaggle:
   ```
   notebooks/full_saree_training_150k.ipynb
   ```
   Upload via: *Kaggle → New Notebook → File → Import Notebook*

3. **Configure Kaggle secrets** (Add-ons → Secrets):
   - `KAGGLE_USERNAME` — your Kaggle username
   - `KAGGLE_KEY` — from https://www.kaggle.com/settings → API → Create New Token

4. **Set runtime:**
   - Accelerator: **GPU T4 × 2** (or P100)
   - Internet access: **On**

5. **Update dataset slugs** in Cell 3 with your actual Kaggle dataset slugs:
   ```python
   DATASETS = [
       {'slug': 'your-username/saree-dataset',   'category': 'saree_images'},
       {'slug': 'your-username/body-pose-data',  'category': 'body_images'},
       {'slug': 'your-username/fabric-textures', 'category': 'blouse_materials'},
   ]
   ```

6. **Run all cells** (estimated 4–8 hours for 150k images, 20 epochs)

7. **Download outputs** from the Kaggle Output panel:
   - `saree_tryon_model.pth`
   - `model_config.json`
   - `training_report.json`

8. **Deploy weights locally:**
   ```
   sareedrap/
   └── backend/
       └── dataset/
           └── trained_models/
               ├── saree_tryon_model.pth   ← place here
               ├── model_config.json       ← place here
               └── training_report.json    ← optional, for reference
   ```
   ```bash
   # Windows
   copy saree_tryon_model.pth backend\dataset\trained_models\
   copy model_config.json     backend\dataset\trained_models\
   ```

9. **Restart Flask backend:**
   ```bash
   cd backend
   venv\Scripts\python app.py
   ```
   The startup log will show:
   ```
   ML engine     : available (weights found)
   ```
   And `/api/render/engine-status` will return `engine: ml_model`.

---

### How the Engine Chooses ML vs Geometric

```
backend/dataset/trained_models/saree_tryon_model.pth
        │
        ├── EXISTS and valid (stub=False)  →  ML inference (SareeTryOnModel, 113M params)
        └── ABSENT or stub=True            →  Geometric fallback (OpenCV draping engine)
```

Check via:
```
GET /api/render/engine-status
```

Response:
```json
{
  "ml_model_available": true,
  "engine": "ml_model",
  "model_loaded": true,
  "is_stub": false,
  "is_prototype": false,
  "param_count_m": 113.53,
  "device": "cpu"
}
```

---

### ML Pipeline Architecture

```
person RGB (3)  ─────────────────────────────────────────────┐
saree  RGB (3)  → SareeWarpingNetwork (GMM + TPS)             │
blouse RGB (3)  → SareeWarpingNetwork (GMM + TPS)             │
pose heatmap(18)─────────────────────────────────────────────├→ TryOnGenerator (U-Net 6-level)
seg mask    (1) ─────────────────────────────────────────────│     → rendered (3, 256×192)
style embed (1) ─────────────────────────────────────────────┘     → alpha   (1, 256×192)
```

### Key Files

| File | Purpose |
|---|---|
| `backend/ai_engine/ml_models.py` | PyTorch model classes (SareeTryOnModel, 113M params) |
| `backend/ai_engine/ml_saree_inference.py` | Inference wrapper + weight verification |
| `backend/routes/train_routes.py` | Training routes — prototype + Kaggle gate |
| `notebooks/full_saree_training_150k.ipynb` | Full 150k Kaggle GPU training notebook |
| `backend/dataset/trained_models/saree_tryon_model.pth` | Trained weights (place here after Kaggle) |
| `backend/dataset/trained_models/model_config.json` | Model hyperparameters |

---

## License

MIT License — Built for educational and demonstration purposes.
