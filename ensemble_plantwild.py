"""Evaluate ensembles of the four PlantWild models on the test set.

Saves:
  model/plantwild_test_probs.npz        (per-model + ensemble probabilities)
  model/plantwild_ensemble_results.json (metrics, per-class, confusions)

Run with the TF venv (TF + torch are both installed there):
  venv\\Scripts\\python.exe ensemble_plantwild.py
"""

import json
import os
import sys
from pathlib import Path

# Must be set before importing dinov2_common (it reads it at import time).
os.environ.setdefault("PDD_DATASET", "plantwild")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

BASE = Path(r"C:\Plant_disease_detection")
TEST_DIR = BASE / "plantwild" / "split" / "test"
MODEL_DIR = BASE / "model"

TF_MODELS = {
    "Basic CNN": (MODEL_DIR / "plantwild_simple_cnn.keras", False),   # raw 0-255
    "ResNet18": (MODEL_DIR / "plantwild_resnet18.keras", True),
    "MobileNetV2": (MODEL_DIR / "plantwild_mobilenetv2.keras", True),
}

sys.path.insert(0, str(BASE / "dinov2_experiment"))
import dinov2_common as common  # noqa: E402


def metrics(y_true, y_pred, n):
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n), zero_division=0
    )
    return {
        "test_accuracy": float(np.mean(y_true == y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "correct": int(np.sum(y_true == y_pred)),
        "total": int(len(y_true)),
        "per_class": [
            {"class_index": i, "class_name": None,
             "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f1[i]), "support": int(s[i])}
            for i in range(n)
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(n)).tolist(),
        "top_confusions": [],
    }


def add_top_confusions(m, names):
    cm = np.array(m["confusion_matrix"])
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((int(cm[i][j]), names[i], names[j]))
    pairs.sort(reverse=True)
    m["top_confusions"] = [
        {"count": c, "true": t, "predicted": p} for c, t, p in pairs[:15]
    ]
    for c in m["per_class"]:
        c["class_name"] = names[c["class_index"]]


def main():
    probs_path = MODEL_DIR / "plantwild_test_probs.npz"

    if probs_path.exists():
        data = np.load(probs_path)
        y_true = data["y_true"]
        names = list(common.class_names())
        n = len(names)
        probs = {
            "Basic CNN": data["probs_basic_cnn"],
            "ResNet18": data["probs_resnet18"],
            "MobileNetV2": data["probs_mobilenetv2"],
            "DINOv2 + logreg": data["probs_dinov2"],
        }
        print(f"Loaded cached test probabilities for {len(y_true)} images")
        return evaluate(names, n, y_true, probs)

    # ---- ground truth + TF probabilities ----
    ds = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR), image_size=(224, 224), batch_size=32,
        shuffle=False, label_mode="int")
    names = ds.class_names
    n = len(names)
    ds_norm = ds.map(lambda x, y: (x / 255.0, y))
    y_true = np.concatenate([y.numpy() for _, y in ds])
    print(f"Test set: {len(y_true)} images, {n} classes")

    probs = {}
    for name, (path, needs_norm) in TF_MODELS.items():
        print(f"Predicting with {name} ...")
        model = tf.keras.models.load_model(path)
        src = ds_norm if needs_norm else ds
        probs[name] = np.concatenate(
            [model.predict(x, verbose=0) for x, _ in src])

    # ---- DINOv2 probabilities from cached features ----
    print("Predicting with DINOv2 + logreg ...")
    import joblib
    cache = common.feature_cache_paths("test", common.BACKBONE_NAME, common.DINOV2_CROP)
    X = np.load(cache["features"]).astype(np.float32)
    y_dino = np.load(cache["labels"]).astype(np.int64)
    if not np.array_equal(y_true, y_dino):
        raise SystemExit("DINOv2 cached labels do not match TF test labels!")
    pipeline = joblib.load(common.artifact_path("dinov2_logreg.joblib"))
    probs["DINOv2 + logreg"] = pipeline.predict_proba(X).astype(np.float32)

    if set(probs) != set(["Basic CNN", "ResNet18", "MobileNetV2", "DINOv2 + logreg"]):
        raise SystemExit("Unexpected model set")

    np.savez_compressed(
        probs_path,
        y_true=y_true,
        probs_basic_cnn=probs["Basic CNN"],
        probs_resnet18=probs["ResNet18"],
        probs_mobilenetv2=probs["MobileNetV2"],
        probs_dinov2=probs["DINOv2 + logreg"],
    )
    print(f"Cached: {probs_path}")
    return evaluate(names, n, y_true, probs)


def evaluate(names, n, y_true, probs):
    # ---- validation accuracies -> weights ----
    def val_acc(fname, log_name):
        d = json.loads((MODEL_DIR / fname).read_text(encoding="utf-8"))
        v = d.get("best_val_accuracy")
        if v:
            return float(v)
        # Older results files never stored it -> recover from the run log.
        from generate_plantwild_figures import parse_history
        hist = parse_history(BASE / "logs" / log_name)
        vals = [h["val_acc"] for h in hist if h["val_acc"] is not None]
        return float(max(vals))

    val = {
        "Basic CNN": val_acc("plantwild_simple_cnn_results.json",
                             "pw_simple_cnn_20261018_204740.txt"),
        "ResNet18": val_acc("plantwild_resnet18_results.json",
                            "pw_resnet18_20261018_204740.txt"),
        "MobileNetV2": val_acc("plantwild_mobilenetv2_results.json",
                               "pw_mobilenetv2_20261018_204740.txt"),
        "DINOv2 + logreg": json.loads(
            (BASE / "dinov2_experiment" / "results" /
             "plantwild_dinov2_validation.json").read_text(encoding="utf-8")
        )["val_accuracy"],
    }
    w = np.array([val[k] for k in probs], dtype=float)
    w = w / w.sum()

    P = np.stack([probs[k] for k in probs], axis=0)  # (4, N, C)

    ensembles = {
        "mean_all": P.mean(axis=0),
        "weighted_val": np.tensordot(w, P, axes=1),
        "dinov2_plus_mobilenet": 0.5 * probs["DINOv2 + logreg"] + 0.5 * probs["MobileNetV2"],
    }

    results = {
        "dataset": "PlantWild (34 classes)",
        "split": "test",
        "class_names": names,
        "model_val_accuracy": val,
        "weights": {k: float(x) for k, x in zip(probs, w)},
        "individual": {},
        "ensembles": {},
    }

    for name, p in probs.items():
        m = metrics(y_true, np.argmax(p, axis=1), n)
        add_top_confusions(m, names)
        results["individual"][name] = m
        print(f"  {name:<18} acc={m['test_accuracy']*100:.2f}% "
              f"macroF1={m['macro_f1']*100:.2f}%")

    for name, p in ensembles.items():
        m = metrics(y_true, np.argmax(p, axis=1), n)
        add_top_confusions(m, names)
        results["ensembles"][name] = m
        print(f"  ENSEMBLE {name:<22} acc={m['test_accuracy']*100:.2f}% "
              f"macroF1={m['macro_f1']*100:.2f}%")

    out = MODEL_DIR / "plantwild_ensemble_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    np.savez_compressed(
        MODEL_DIR / "plantwild_test_probs.npz",
        y_true=y_true,
        probs_basic_cnn=probs["Basic CNN"],
        probs_resnet18=probs["ResNet18"],
        probs_mobilenetv2=probs["MobileNetV2"],
        probs_dinov2=probs["DINOv2 + logreg"],
        probs_mean_all=ensembles["mean_all"],
        probs_weighted=ensembles["weighted_val"],
    )
    print(f"\nSaved: {out}")
    print(f"Saved: {MODEL_DIR / 'plantwild_test_probs.npz'}")
    return results


if __name__ == "__main__":
    main()
