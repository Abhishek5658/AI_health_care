# -*- coding: utf-8 -*-
# HealthAI: serve static pages + Brain & Lungs inference APIs
# Brain: Xception (4 classes) — labels baked in
# Lungs: EfficientNetV2 — auto-pick model, force 5-class names when output=5

import io, os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications import xception as xcep
from tensorflow.keras.applications import efficientnet_v2 as effv2

# ---------- Paths ----------
SERVER_DIR = Path(__file__).resolve().parent
APP_ROOT   = SERVER_DIR.parent                         # .../web application
INDEX_HTML = APP_ROOT / "index.html"

# Brain model (Xception)
BRAIN_MODEL_PATH = APP_ROOT / "brain" / "artifacts" / "best_xception.keras"
BRAIN_LABELS     = ["glioma", "meningioma", "no_tumor", "pituitary"]  # baked-in order

# Lung models (EffNetV2 variants)
LUNG_DIR = APP_ROOT / "lung" / "artifacts"
LUNG_CANDIDATES = [
    LUNG_DIR / "best_effnetv2b0.keras",
    LUNG_DIR / "effnetv2b0_final.keras",
    LUNG_DIR / "effnetv2b3_lungs_finetune_best.h5",
]
ENV_LUNG_MODEL  = os.getenv("LUNG_MODEL_PATH")  # optional path override
ENV_LUNG_LABELS = os.getenv("LUNG_LABELS")      # optional comma list

# ---------- Safety checks ----------
assert INDEX_HTML.exists(), f"Missing index.html at {INDEX_HTML}"
assert BRAIN_MODEL_PATH.exists(), f"Missing brain model at {BRAIN_MODEL_PATH}"

# ---------- GPU-friendly TF ----------
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ---------- Helpers ----------
ALLOWED_EXT = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def allowed_file(name: str) -> bool:
    return Path(name.lower()).suffix in ALLOWED_EXT

def infer_input_hw(model) -> Tuple[int, int]:
    shp = list(model.input_shape)
    if len(shp) == 4:  # (None, H, W, C)
        return int(shp[1]), int(shp[2])
    if len(shp) == 3:  # (H, W, C)
        return int(shp[0]), int(shp[1])
    return (224, 224)

def infer_num_classes(model) -> int:
    out = model.output_shape
    if isinstance(out, (list, tuple)) and isinstance(out[0], (list, tuple)):
        out = out[0]
    return int(out[-1])

def forced_five_labels() -> List[str]:
    # Matches your training folders from your message:
    # ['Bacterial Pneumonia','Corona Virus Disease','Normal','Tuberculosis','Viral Pneumonia']
    return ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

def derive_labels(num_classes: int, env_labels: Optional[str]) -> List[str]:
    if env_labels:
        names = [s.strip() for s in env_labels.split(",") if s.strip()]
        if len(names) == num_classes:
            return names
    if num_classes == 5:
        return forced_five_labels()
    if num_classes == 2:
        return ["pneumonia", "normal"]
    return [f"class_{i}" for i in range(num_classes)]

def preprocess_file(file_storage, size: Tuple[int,int], pre_fn) -> np.ndarray:
    img = Image.open(io.BytesIO(file_storage.read()))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    arr = img_to_array(img)
    arr = pre_fn(arr)
    return np.expand_dims(arr, axis=0)

def logits_to_payload(logits: np.ndarray, labels: List[str]) -> Dict:
    probs = logits[0].astype("float64")
    probs = probs / (probs.sum() + 1e-8)
    k = int(np.argmax(probs))
    return {
        "top": labels[k],
        "conf": float(probs[k]),
        "probs": {labels[i]: float(probs[i]) for i in range(len(labels))}
    }

# ---------- Load models once ----------
# Brain (Xception)
BRAIN_MODEL = tf.keras.models.load_model(str(BRAIN_MODEL_PATH), compile=False)
BRAIN_PRE   = xcep.preprocess_input
BRAIN_SIZE  = infer_input_hw(BRAIN_MODEL) or (299, 299)

# Lungs (EffNetV2) — select model with MOST classes; force 5 names if 5 outputs
LUNG_MODEL = None
LUNG_PRE   = effv2.preprocess_input
LUNG_SIZE  = (224, 224)
LUNG_LABELS: List[str] = []
LUNG_MODEL_PATH: Optional[Path] = None
LUNG_NUM_CLASSES: Optional[int] = None

def select_lung_model() -> None:
    global LUNG_MODEL, LUNG_MODEL_PATH, LUNG_SIZE, LUNG_LABELS, LUNG_NUM_CLASSES
    candidates: List[Path] = []
    if ENV_LUNG_MODEL:
        p = Path(ENV_LUNG_MODEL)
        if not p.is_absolute():
            p = (APP_ROOT / p).resolve()
        if p.exists():
            candidates.append(p)
    candidates += [p for p in LUNG_CANDIDATES if p.exists()]

    if not candidates:
        print("[LUNG] No model files found in lung/artifacts/")
        return

    best: Optional[Tuple[int, Path]] = None
    for p in candidates:
        try:
            print(f"[LUNG] Probing {p.name} ...")
            m = tf.keras.models.load_model(str(p), compile=False)
            num = infer_num_classes(m)
            print(f"[LUNG]   -> classes: {num}")
            if (best is None) or (num > best[0]):
                best = (num, p)
            del m
        except Exception as e:
            print(f"[LUNG]   !! Failed to load {p.name}: {e}")

    if best is None:
        print("[LUNG] Could not load any candidate successfully.")
        return

    LUNG_NUM_CLASSES, LUNG_MODEL_PATH = best
    LUNG_MODEL = tf.keras.models.load_model(str(LUNG_MODEL_PATH), compile=False)
    LUNG_SIZE  = infer_input_hw(LUNG_MODEL) or (224, 224)
    LUNG_LABELS = derive_labels(LUNG_NUM_CLASSES, ENV_LUNG_LABELS)
    print(f"[LUNG] Selected: {LUNG_MODEL_PATH.name} with {LUNG_NUM_CLASSES} classes")
    print(f"[LUNG] Labels: {LUNG_LABELS}")

select_lung_model()

# ---------- Flask ----------
app = Flask(__name__, static_folder=None)
CORS(app)

# ---- Health ----
@app.get("/health")
def health():
    return {
        "ok": True,
        "brain": {"model": BRAIN_MODEL_PATH.name, "input_hw": BRAIN_SIZE, "labels": BRAIN_LABELS},
        "lung": {
            "model": (LUNG_MODEL_PATH.name if LUNG_MODEL_PATH else None),
            "loaded": bool(LUNG_MODEL),
            "input_hw": LUNG_SIZE,
            "num_classes": LUNG_NUM_CLASSES,
            "labels": LUNG_LABELS,
        },
    }

# ---- Brain API ----
@app.post("/api/brain/predict")
def api_brain():
    if "image" not in request.files:
        return jsonify({"error": "Form field 'image' is required."}), 400
    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "Please upload an image (.jpg/.png/...)"}), 400
    try:
        x = preprocess_file(f, BRAIN_SIZE, BRAIN_PRE)
        logits = BRAIN_MODEL.predict(x, verbose=0)
        return jsonify(logits_to_payload(logits, BRAIN_LABELS))
    except Exception as e:
        return jsonify({"error": f"Inference error: {e}"}), 500

# ---- Lungs API ----
@app.post("/api/lung/predict")
def api_lung():
    if LUNG_MODEL is None:
        return jsonify({"error": "Lung model not loaded. Place a .keras/.h5 in lung/artifacts/"}), 501
    if "image" not in request.files:
        return jsonify({"error": "Form field 'image' is required."}), 400
    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "Please upload an image (.jpg/.png/...)"}), 400
    try:
        x = preprocess_file(f, LUNG_SIZE, LUNG_PRE)
        logits = LUNG_MODEL.predict(x, verbose=0)
        return jsonify(logits_to_payload(logits, LUNG_LABELS))
    except Exception as e:
        return jsonify({"error": f"Inference error: {e}"}), 500

# ---------- Static pages ----------
@app.get("/")
def root():
    return send_file(INDEX_HTML)

@app.route("/<path:relpath>")
def any_static(relpath: str):
    target = (APP_ROOT / relpath).resolve()
    if APP_ROOT not in target.parents and target != APP_ROOT:
        abort(404)
    if target.is_dir():
        index = target / "index.html"
        if index.exists():
            return send_file(index)
        abort(404)
    if target.exists():
        return send_file(target)
    abort(404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"[HealthAI] Serving from: {APP_ROOT}")
    app.run(host="0.0.0.0", port=port, debug=False)
