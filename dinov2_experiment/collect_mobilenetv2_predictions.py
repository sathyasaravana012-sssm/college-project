"""Collect MobileNetV2 predictions for the comparison (run in the TF venv).

This script NEVER runs in the isolated DINOv2 environment, because that
environment has no TensorFlow. Run it with the project's existing interpreter:

    # from C:\\Plant_disease_detection
    .\\venv\\Scripts\\python.exe dinov2_experiment\\collect_mobilenetv2_predictions.py

It only reads images and writes prediction files under
``dinov2_experiment/results``. It does not modify, retrain or overwrite the
existing models, splits or evaluation results.

The preprocessing here is copied from the existing, already-validated scripts
(``evaluate_mobilenetv2_plantvillage.py`` and
``evaluate_mobilenetv2_plantdoc.py``) so the baseline is measured exactly the
way it was originally measured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dinov2_common as common

# Models available in config.BASELINE_MODELS. "basic" is plant_disease_model.keras
# (the MobileNetV2 checkpoint with 93.98% PlantVillage test accuracy).
MODEL_FILES = {
    "basic": "plant_disease_model.keras",
    "mobilenetv2": "mobilenetv2_plant_disease.keras",
    "basic_plantdoc": "basic_plantdoc.keras",
    "mobilenetv2_plantdoc": "mobilenetv2_plantdoc.keras",
}


def _load_model(model_key: str):
    import tensorflow as tf

    path = common.PROJECT_ROOT / "model" / MODEL_FILES[model_key]
    if not path.exists():
        raise SystemExit(f"Model not found: {path}")
    print(f"Loading {model_key}: {path.name}")
    return tf.keras.models.load_model(str(path))


def _predict_batches(model, images: np.ndarray, batch_size: int) -> np.ndarray:
    probs = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size].astype(np.float32) / 255.0
        probs.append(model.predict(chunk, verbose=0))
        print(f"  predicted {min(start + batch_size, len(images))}/{len(images)}")
    return np.concatenate(probs, axis=0)


def _load_images(records, resize_to):
    from PIL import Image

    images = []
    labels = []
    paths = []
    for path, label in records:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(resize_to, Image.BILINEAR)
                images.append(np.asarray(img))
            labels.append(label)
            paths.append(path)
        except Exception as exc:
            print(f"[warn] skipping unreadable image {path}: {exc}")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.int64), paths


def collect(dataset: str, model_key: str, batch_size: int) -> None:
    model = _load_model(model_key)

    if dataset == "plantvillage":
        records = common.plantvillage_records(common.TEST_DIR)
    elif dataset == "realworld":
        records = common.plantdoc_records(common.REAL_WORLD_DIR)
    else:
        raise SystemExit(f"Unknown dataset: {dataset}")

    print(f"Dataset '{dataset}': {len(records)} images")
    images, y_true, paths = _load_images(records, (224, 224))
    probs = _predict_batches(model, images, batch_size)
    y_pred = probs.argmax(axis=1)

    out = common.RESULTS_DIR / f"mobilenetv2_{dataset}_preds.npz"
    np.savez(
        out,
        y_true=y_true,
        y_pred=y_pred,
        probs=probs.astype(np.float32),
        paths=np.asarray(paths, dtype=object),
        model_key=np.asarray(model_key),
    )
    accuracy = float((y_pred == y_true).mean())
    print(f"Saved {out}  (accuracy={accuracy*100:.2f}%)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["plantvillage", "realworld"],
                        choices=["plantvillage", "realworld"])
    parser.add_argument("--model-key", default="basic",
                        choices=sorted(MODEL_FILES))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    for ds in args.datasets:
        collect(ds, args.model_key, args.batch_size)
