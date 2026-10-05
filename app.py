import io
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.utils import secure_filename

import config

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("PLANT_APP_SECRET_KEY", "dev-key-change-me")

app.config["TEMPLATES_AUTO_RELOAD"] = True

_MODEL_CACHE = {}


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _central_crop_resize(image: Image.Image, size: Tuple[int, int] = config.IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    target = min(width, height)
    left = (width - target) // 2
    top = (height - target) // 2
    right = left + target
    bottom = top + target
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(size)


def _load_disease_info():
    path = config.DISEASE_INFO_PATH
    if not path.exists():
        raise FileNotFoundError(f"Disease info not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    missing = [c for c in config.CLASS_NAMES if c not in data]
    if missing:
        raise ValueError(f"Disease info missing for classes: {missing}")
    return data


def _warmup_models():
    for key, path in config.BASELINE_MODELS.items():
        try:
            _get_model(key, path)
        except Exception:
            pass


@app.before_request
def _ensure_warmup():
    if not _MODEL_CACHE:
        _warmup_models()


def _get_model(name: str, path: Path):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = tf.keras.models.load_model(path)
    if model.output_shape[-1] != config.NUM_CLASSES:
        raise ValueError(
            f"Model {name} outputs {model.output_shape[-1]} classes, "
            f"expected {config.NUM_CLASSES}"
        )
    _MODEL_CACHE[name] = model
    return model


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=config.BASELINE_MODELS.keys(),
        default_model="basic",
        class_names=config.CLASS_NAMES,
    )


@app.route("/health")
def health():
    loaded = list(_MODEL_CACHE.keys())
    return jsonify({"status": "ok", "loaded_models": loaded})


@app.route("/predict", methods=["POST"])
def predict():
    disease_info = _load_disease_info()

    if "image" not in request.files:
        flash("No image uploaded. Please select an image file and try again.", "error")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("No image selected. Please choose an image file.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            "error",
        )
        return redirect(url_for("index"))

    model_name = request.form.get("model", "basic").lower()

    if model_name not in config.BASELINE_MODELS:
        flash(f"Unknown model. Choose from {', '.join(sorted(config.BASELINE_MODELS))}", "error")
        return redirect(url_for("index"))

    try:
        model = _get_model(model_name, config.BASELINE_MODELS[model_name])
    except Exception as e:
        flash(f"Failed to load model '{model_name}': {e}", "error")
        return redirect(url_for("index"))

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        flash("Unable to read the uploaded image. Is it a valid image file?", "error")
        return redirect(url_for("index"))

    try:
        image = _central_crop_resize(image)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
    except Exception as e:
        flash(f"Failed to preprocess the image: {e}", "error")
        return redirect(url_for("index"))

    try:
        predictions = model.predict(image_array, verbose=0)[0]
    except Exception as e:
        flash(f"Prediction failed: {e}", "error")
        return redirect(url_for("index"))

    top_indices = np.argsort(predictions)[-5:][::-1]
    top_results = []

    for idx in top_indices:
        class_name = config.CLASS_NAMES[idx]
        prob = float(predictions[idx])
        info = disease_info.get(class_name, {})
        top_results.append(
            {
                "class_index": int(idx),
                "class_name": class_name,
                "display_name": info.get("display_name") or class_name,
                "plant": info.get("plant") or "",
                "is_healthy": bool(info.get("is_healthy", False)),
                "cause": info.get("cause"),
                "treatment": info.get("treatment"),
                "confidence": prob * 100.0,
            }
        )

    best = top_results[0] if top_results else None

    return render_template(
        "index.html",
        models=config.BASELINE_MODELS.keys(),
        default_model=model_name,
        class_names=config.CLASS_NAMES,
        result=True,
        best=best,
        top_results=top_results,
        uploaded_filename=secure_filename(file.filename),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)