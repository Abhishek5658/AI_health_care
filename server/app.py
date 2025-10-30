# server/app.py
import os, io, time, json, traceback
from typing import Tuple
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------- Paths -----------------
HERE = os.path.dirname(os.path.abspath(__file__))      # .../server
ROOT = os.path.abspath(os.path.join(HERE, ".."))       # repo root

BRAIN_MODEL_PATH = os.path.join(ROOT, "brain", "artifacts", "best_xception.keras")
LUNG_MODEL_PATH  = os.path.join(ROOT, "lung",  "artifacts", "effnetv2b0_final.keras")

# Serve ALL static files (index.html, brain.html, lung.html, /assets/*) from repo root
app = Flask(__name__, static_folder=ROOT, static_url_path="")
CORS(app)

# ----------- Class names -----------
BRAIN_CLASSES_FALLBACK = ["glioma", "meningioma", "no_tumor", "pituitary"]
def read_brain_classes() -> list:
    txt = os.path.join(ROOT, "brain", "artifacts", "classes.txt")
    if os.path.isfile(txt):
        with open(txt, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    return BRAIN_CLASSES_FALLBACK

BRAIN_CLASSES = read_brain_classes()
LUNG_CLASSES = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia",
]

# ----------- Lazy TF / models -----------
_tf = _xcep = _effv2 = None
_brain_model = _lung_model = None
_brain_img_size = (299, 299)
_lung_img_size  = (224, 224)

def _import_tf():
    global _tf, _xcep, _effv2
    if _tf is None:
        import tensorflow as tf
        from tensorflow.keras.applications import xception as xcep
        from tensorflow.keras.applications import efficientnet_v2 as effv2
        _tf, _xcep, _effv2 = tf, xcep, effv2
        try:
            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    return _tf

def _ensure_brain():
    global _brain_model, _brain_img_size
    if _brain_model is None:
        tf = _import_tf()
        _brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH, compile=False)
        ish = getattr(_brain_model, "input_shape", None)
        if isinstance(ish, tuple) and len(ish) == 4 and ish[1] and ish[2]:
            _brain_img_size = (int(ish[1]), int(ish[2]))

def _ensure_lung():
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
    _import_tf()
    arr = _pil_to_array(pil_img, _brain_img_size)
    arr = _xcep.preprocess_input(arr)
    x = np.expand_dims(arr, 0)
    probs = _brain_model.predict(x, verbose=0)[0]
    probs = probs / (probs.sum() + 1e-8)
    k = int(np.argmax(probs))
    return BRAIN_CLASSES[k], float(probs[k])

def _predict_lung(pil_img: Image.Image):
    _ensure_lung()
    _import_tf()
    arr = _pil_to_array(pil_img, _lung_img_size)
    arr = _effv2.preprocess_input(arr)
    x = np.expand_dims(arr, 0)
    probs = _lung_model.predict(x, verbose=0)[0]
    probs = probs / (probs.sum() + 1e-8)
    k = int(np.argmax(probs))
    return LUNG_CLASSES[k], float(probs[k])

# ----------- Education blurbs -----------
LUNG_EDU = {
    "Bacterial Pneumonia": "Bacterial infection of lung; X-ray may show lobar consolidation. Care: clinician-guided antibiotics, fever control, hydration, follow-up if not improving.",
    "Corona Virus Disease": "Viral pneumonia; ground-glass opacities common. Management depends on severity—home isolation & symptomatic care for mild, seek care for antivirals/oxygen if hypoxic.",
    "Normal": "No obvious abnormality on X-ray. If symptoms persist (fever, cough, breathlessness), consult a clinician for further evaluation.",
    "Tuberculosis": "Mycobacterial infection; upper-lobe infiltrates/cavities can appear. Requires supervised multi-drug therapy (e.g., DOTS) and adherence monitoring.",
    "Viral Pneumonia": "Viral lung infection; often diffuse interstitial pattern. Usually supportive care (rest, fluids, antipyretics); antibiotics only if bacterial superinfection suspected.",
}

# ----------- Health & API -----------
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "brain_model": os.path.basename(BRAIN_MODEL_PATH),
        "lung_model": os.path.basename(LUNG_MODEL_PATH),
        "brain_classes": BRAIN_CLASSES,
        "lung_classes": LUNG_CLASSES,
        "status": "ready"
    })

def _extract_image():
    if "image" not in request.files:
        raise ValueError("No 'image' file provided.")
    f = request.files["image"]
    if not f or f.filename == "":
        raise ValueError("Empty file.")
    return Image.open(io.BytesIO(f.read()))

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
            "educational": {"title": f"Predicted: {label}",
                            "note": "AI triage output. Final diagnosis is by a clinician."}
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
            "educational": {"title": f"Predicted: {label}",
                            "about": LUNG_EDU.get(label, ""),
                            "disclaimer": "Assistive triage only; consult a clinician."}
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 400

# ----------- Static pages -----------
@app.get("/")
def home():
    # served from ROOT (repo root)
    return app.send_static_file("index.html")

@app.get("/brain")
def brain_page():
    return app.send_static_file("brain.html")

@app.get("/lung")
def lung_page():
    return app.send_static_file("lung.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
