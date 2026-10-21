"""PlantWild plant-disease detector — Flask website.

Ensemble of the four trained PlantWild models:
  Basic CNN, ResNet18, MobileNetV2 (TensorFlow) + DINOv2 ViT-S/14 with
  logistic regression (PyTorch).  Probabilities are combined with the
  validation-accuracy weights from model/plantwild_ensemble_results.json
  (the equal-mean ensemble was measurably worse: 62.31% vs 73.87%).

Run:
    venv\\Scripts\\python.exe website\\app.py
then open http://127.0.0.1:5000
"""

import io
import json
import os
import sys
import uuid
from pathlib import Path

# Must happen before dinov2_common is imported (it reads it at import time).
os.environ.setdefault("PDD_DATASET", "plantwild")

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "dinov2_experiment"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

import dinov2_common as common  # noqa: E402

MODEL_DIR = BASE / "model"
UPLOAD_DIR = Path(__file__).resolve().parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR = BASE / "plantwild" / "split" / "test"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = os.environ.get("PLANTWILD_FLASK_KEY", "dev-only-demo-key")

# --------------------------------------------------------------------------
# Load results / weights
# --------------------------------------------------------------------------
RESULTS = json.loads(
    (MODEL_DIR / "plantwild_ensemble_results.json").read_text(encoding="utf-8")
)
CLASS_NAMES = RESULTS["class_names"]
WEIGHTS = RESULTS["weights"]           # {"Basic CNN": .., "ResNet18": .., ..}
MODEL_ORDER = ["Basic CNN", "ResNet18", "MobileNetV2", "DINOv2 + logreg"]

# --------------------------------------------------------------------------
# Load models
# --------------------------------------------------------------------------
print("[web] loading TensorFlow models ...")
TF_MODELS = {
    "Basic CNN": tf.keras.models.load_model(MODEL_DIR / "plantwild_simple_cnn.keras"),
    "ResNet18": tf.keras.models.load_model(MODEL_DIR / "plantwild_resnet18.keras"),
    "MobileNetV2": tf.keras.models.load_model(MODEL_DIR / "plantwild_mobilenetv2.keras"),
}
RAW_INPUT_MODELS = {"Basic CNN"}  # contains its own Rescaling(1/255) layer

print("[web] loading DINOv2 backbone + classifier ...")
import joblib  # noqa: E402

DINO_BACKBONE = common.load_backbone(device="cpu")
DINO_PIPELINE = joblib.load(common.artifact_path("dinov2_logreg.joblib"))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_image(img: Image.Image):
    """img -> per-model probs + ensembles. Returns a dict."""
    img = img.convert("RGB")

    arr = tf.keras.utils.img_to_array(img.resize((224, 224)))
    probs = {}
    for name, model in TF_MODELS.items():
        x = arr if name in RAW_INPUT_MODELS else arr / 255.0
        probs[name] = model.predict(np.expand_dims(x, 0), verbose=0)[0]

    tensor = common.preprocess_pil(img)
    feats = common.extract_features_batch(DINO_BACKBONE, tensor.unsqueeze(0), "cpu")
    probs["DINOv2 + logreg"] = DINO_PIPELINE.predict_proba(feats)[0]

    P = np.stack([probs[k] for k in MODEL_ORDER], axis=0)
    w = np.array([WEIGHTS[k] for k in MODEL_ORDER], dtype=float)

    weighted = np.tensordot(w, P, axes=1)
    equal = P.mean(axis=0)

    def top5(p):
        order = np.argsort(p)[::-1][:5]
        return [{"class": CLASS_NAMES[int(i)], "prob": float(p[int(i)])}
                for i in order]

    return {
        "per_model": [
            {
                "name": name,
                "top1": top5(probs[name])[0],
                "top5": top5(probs[name]),
            }
            for name in MODEL_ORDER
        ],
        "ensemble_weighted": {
            "top1": top5(weighted)[0],
            "top5": top5(weighted),
        },
        "ensemble_equal": {
            "top1": top5(equal)[0],
            "top5": top5(equal),
        },
    }


# --------------------------------------------------------------------------
# Sample images from the held-out test set
# --------------------------------------------------------------------------
SAMPLES = []
for _cls in sorted(p.name for p in TEST_DIR.iterdir() if p.is_dir()):
    files = sorted(f.name for f in (TEST_DIR / _cls).iterdir()
                   if f.suffix.lower() in common.IMAGE_EXTENSIONS)
    if files:
        SAMPLES.append({"class": _cls, "file": files[0]})
SAMPLES = SAMPLES[:12]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
def _index_ctx(error=None):
    best = max(RESULTS["ensembles"],
               key=lambda k: RESULTS["ensembles"][k]["test_accuracy"])
    return dict(
        samples=SAMPLES,
        error=error,
        individual_models=RESULTS["individual"],
        ensemble_acc=f"{RESULTS['ensembles'][best]['test_accuracy']*100:.2f}%",
    )


@app.route("/")
def index():
    return render_template("index.html", **_index_ctx())


@app.route("/sample_image")
def sample_image():
    cls = request.args.get("cls", "")
    name = request.args.get("file", "")
    root = (TEST_DIR / cls / name).resolve()
    if not str(root).startswith(str(TEST_DIR.resolve())) or not root.is_file():
        abort(404)
    return send_file(root)


def _resolve_input():
    """Returns (PIL image, display_url, label) or raises ValueError."""
    f = request.files.get("image")
    if f and f.filename:
        raw = f.read()
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception as exc:
            raise ValueError(f"Could not read that image file: {exc}")
        suffix = Path(f.filename).suffix.lower() or ".png"
        saved = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        saved.write_bytes(raw)
        return img, url_for("static", filename=f"uploads/{saved.name}"), f.filename

    cls = request.form.get("sample_class", "")
    name = request.form.get("sample_file", "")
    if cls and name:
        root = (TEST_DIR / cls / name).resolve()
        if str(root).startswith(str(TEST_DIR.resolve())) and root.is_file():
            img = Image.open(root)
            img.load()
            return img, url_for("sample_image", cls=cls, file=name), f"{cls}/{name}"

    raise ValueError("Upload an image or pick one of the sample photos.")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img, display_url, label = _resolve_input()
    except ValueError as exc:
        return render_template("index.html", **_index_ctx(str(exc))), 400

    out = predict_image(img)
    return render_template(
        "result.html",
        image_url=display_url,
        label=label,
        result=out,
        weights=WEIGHTS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        img, _, label = _resolve_input()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    out = predict_image(img)
    out["label"] = label
    return jsonify(out)


@app.route("/report")
def report():
    ens = RESULTS["ensembles"]
    ind = RESULTS["individual"]
    best_name = max(ens, key=lambda k: ens[k]["test_accuracy"])
    figures = [
        ("confusion_basic_cnn.png",
         f"Basic CNN — {ind['Basic CNN']['test_accuracy']*100:.2f}% acc"),
        ("confusion_resnet18.png",
         f"ResNet18 — {ind['ResNet18']['test_accuracy']*100:.2f}% acc"),
        ("confusion_mobilenetv2.png",
         f"MobileNetV2 — {ind['MobileNetV2']['test_accuracy']*100:.2f}% acc"),
        ("confusion_dinov2.png",
         f"DINOv2 + logreg — {ind['DINOv2 + logreg']['test_accuracy']*100:.2f}% acc"),
        ("progress_models.png",
         "Training progress — the three TensorFlow models (10 epochs, CPU)"),
        ("progress_dinov2.png",
         "DINOv2 head tuning — logistic-regression C sweep (no epochs)"),
    ]
    return render_template(
        "report.html",
        individual=ind,
        ensembles=ens,
        best_name=best_name,
        weights=WEIGHTS,
        figures=figures,
        test_total=ind["Basic CNN"]["total"],
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": len(CLASS_NAMES),
                    "models": MODEL_ORDER})


if __name__ == "__main__":
    print("[web] starting on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
