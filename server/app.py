# server/app.py
import os, io, time, json, traceback
from typing import Optional, Tuple
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=None)
CORS(app)

# --- Paths (relative to repo root) ---
BRAIN_MODEL_PATH = os.path.join("brain", "artifacts", "best_xception.keras")
LUNG_MODEL_PATH  = os.path.join("lung",  "artifacts", "effnetv2b0_final.keras")

# --- Class names (no classes.txt required; fallback if missing) ---
BRAIN_CLASSES_FALLBACK = ["glioma", "meningioma", "no_tumor", "pituitary"]
LUNG_CLASSES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]

# Try to read brain classes.txt if present; else fallback
def read_brain_classes() -> list:
    txt = os.path.join("brain", "artifacts", "classes.txt")
    if os.path.isfile(txt):
        with open(txt, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    return BRAIN_CLASSES_FALLBACK

BRAIN_CLASSES = read_brain_classes()

# Globals to hold loaded models once (lazy)
_tf = None
_xcep = None
_effv2 = None
_brain_model = None
_lung_model = None
_brain_img_size = (299, 299)  # Xception default
_lung_img_size = (224, 224)   # EffNetV2B0 default

def _import_tf():
    """Import TensorFlow only when needed."""
    global _tf, _xcep, _effv2
    if _tf is None:
        import tensorflow as tf  # heavy import; avoid at module import time
        from tensorflow.keras.applications import xception as xcep
        from tensorflow.keras.applications import efficientnet_v2 as effv2
        _tf, _xcep, _effv2 = tf, xcep, effv2
        # be polite with memory on small instances
        try:
            for g in _tf.config.list_physical_devices('GPU'):
                _tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    return _tf

def _ensure_brain():
    """Load brain model lazily."""
    global _brain_model, _brain_img_size
    if _brain_model is None:
        tf = _import_tf()
        _brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH, compile=False)
        # Try to infer input size
        ish = getattr(_brain_model, "input_shape", None)
        if isinstance(ish, tuple) and len(ish) == 4 and ish[1] and ish[2]:
            _brain_img_size = (int(ish[1]), int(ish[2]))

def _ensure_lung():
    """Load lung model lazily."""
    global _lung_model, _lung_img_size
    if _lung_model is None:
        tf = _import_tf()
        _lung_model = tf.keras.models.load_model(LUNG_MODEL_PATH, compile=False)
        ish = getattr(_lung_model, "input_shape", None)
        if isinstance(ish, tuple) and len(ish) == 4 and ish[1] and ish[2]:
            _lung_img_size = (int(ish[1]), int(ish[2]))

def _pil_to_array(img: Image.Image, size: Tuple[int,int]) -> np.ndarray:
    arr = img.convert("RGB").resize(size)
    return np.asarray(arr, dtype=np.float32)

def _predict_brain(pil_img: Image.Image):
    _ensure_brain()
    tf = _import_tf()
    # Xception preprocessing [-1,1]
    arr = _pil_to_array(pil_img, _brain_img_size)
    arr = _xcep.preprocess_input(arr)
    x = np.expand_dims(arr, axis=0)
    probs = _brain_model.predict(x, verbose=0)[0]
    probs = probs / (probs.sum() + 1e-8)
    k = int(np.argmax(probs))
    return BRAIN_CLASSES[k], float(probs[k])

def _predict_lung(pil_img: Image.Image):
    _ensure_lung()
    tf = _import_tf()
    # EffNetV2 preprocessing
    arr = _pil_to_array(pil_img, _lung_img_size)
    arr = _effv2.preprocess_input(arr)
    x = np.expand_dims(arr, axis=0)
    probs = _lung_model.predict(x, verbose=0)[0]
    probs = probs / (probs.sum() + 1e-8)
    k = int(np.argmax(probs))
    return LUNG_CLASSES[k], float(probs[k])

# Short education blurbs (2–3 lines) for lung classes
LUNG_EDU = {
    "Bacterial Pneumonia": "Infection of the lungs by bacteria; X-ray may show lobar consolidation. Usual care is physician-guided antibiotics, fever control, hydration, and follow-up film if symptoms persist/worsen.",
    "Corona Virus Disease": "Viral pneumonia pattern; ground-glass opacities common. Management depends on severity—home isolation and symptomatic care for mild cases; seek clinician advice for antivirals/oxygen if hypoxic.",
    "Normal": "No obvious abnormality on the chest X-ray. If symptoms persist (fever, cough, breathlessness), consult a clinician for further evaluation.",
    "Tuberculosis": "Mycobacterial lung infection; upper-lobe infiltrates/cavities can appear. Requires physician-supervised multi-drug therapy (DOTS/NT-TB program) and adherence monitoring.",
    "Viral Pneumonia": "Viral infection of the lungs; often diffuse interstitial changes. Typically supportive care (rest, fluids, antipyretics); antibiotics only if bacterial superinfection is suspected.",
}

@app.get("/health")
def health():
    # Do NOT force model load here—keep boot fast
    return jsonify({
        "ok": True,
        "brain_model": os.path.basename(BRAIN_MODEL_PATH),
        "lung_model": os.path.basename(LUNG_MODEL_PATH),
        "brain_classes": BRAIN_CLASSES,
        "lung_classes": LUNG_CLASSES,
        "status": "ready"
    })

def _extract_image() -> Image.Image:
    if "image" not in request.files:
        raise ValueError("No file field named 'image'.")
    f = request.files["image"]
    if not f or f.filename == "":
        raise ValueError("Empty file.")
    img = Image.open(io.BytesIO(f.read()))
    return img

@app.post("/api/brain/predict")
def api_brain():
    t0 = time.time()
    try:
        img = _extract_image()
        label, conf = _predict_brain(img)
        return jsonify({
            "ok": True,
            "top_class": label,
            "confidence": round(conf, 4),
            "elapsed_sec": round(time.time() - t0, 3),
            "educational": {
                "title": f"Predicted: {label}",
                "note": "AI triage output. Final diagnosis must be made by a qualified clinician."
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 400

@app.post("/api/lung/predict")
def api_lung():
    t0 = time.time()
    try:
        img = _extract_image()
        label, conf = _predict_lung(img)
        return jsonify({
            "ok": True,
            "top_class": label,
            "confidence": round(conf, 4),
            "elapsed_sec": round(time.time() - t0, 3),
            "educational": {
                "title": f"Predicted: {label}",
                "about": LUNG_EDU.get(label, ""),
                "disclaimer": "This is an assistive triage; please consult a clinician for treatment."
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 400

# Serve the simple static HTML (if you’re using single-service deployment)
@app.get("/")
def root():
    # Serve index.html from repo root
    return send_from_directory(".", "index.html")

@app.get("/<path:path>")
def any_static(path):
    if os.path.isfile(path):
        return send_from_directory(".", path)
    return jsonify({"ok": False, "error": "Not found"}), 404

if __name__ == "__main__":
    # Local run
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
