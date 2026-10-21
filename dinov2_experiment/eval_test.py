"""Evaluate the fitted DINOv2 logistic-regression classifier on the TEST split.

Reads the cached test features and the classifier saved by train_classifier.py,
computes the same metrics the TensorFlow scripts report, and writes
``results/<tag>dinov2_test.json``.

    .venv\\Scripts\\python.exe eval_test.py
    set PDD_DATASET=plantwild  (in the shell) to evaluate the PlantWild run
"""

from __future__ import annotations

import sys
import time

import numpy as np

import dinov2_common as common


def _load_split(split: str, backbone: str, crop: int):
    paths = common.feature_cache_paths(split, backbone, crop)
    if not paths["features"].exists() or not paths["labels"].exists():
        raise SystemExit(
            f"Missing cached features for '{split}'. Run extract_features.py first.\n"
            f"  expected: {paths['features']}"
        )
    X = np.load(paths["features"]).astype(np.float32)
    y = np.load(paths["labels"]).astype(np.int64)
    return X, y


def main(argv=None) -> int:
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    backbone = common.BACKBONE_NAME
    crop = common.DINOV2_CROP

    print(f"Dataset: {common.DATASET} ({common.NUM_CLASSES} classes)")
    model_path = common.artifact_path("dinov2_logreg.joblib")
    print(f"Loading classifier: {model_path}")
    pipeline = joblib.load(model_path)

    print("Loading cached test features...")
    X_test, y_test = _load_split("test", backbone, crop)
    print(f"  test: {X_test.shape}")

    start = time.time()
    y_pred = pipeline.predict(X_test)
    elapsed = time.time() - start

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    correct = int(np.sum(y_test == y_pred))

    print("\n" + "=" * 60)
    print(f"FINAL DINOV2 RESULTS ({common.DATASET.upper()} TEST SET)")
    print("=" * 60)
    print(f"Model             : DINOv2 ViT-S/14 + LogisticRegression")
    print(f"Test Accuracy     : {accuracy * 100:.2f}%")
    print(f"Macro Precision   : {precision * 100:.2f}%")
    print(f"Macro Recall      : {recall * 100:.2f}%")
    print(f"Macro F1-score    : {f1 * 100:.2f}%")
    print(f"Correct           : {correct}")
    print(f"Total             : {len(y_test)}")
    print(f"Predict time      : {elapsed:.1f}s")
    print("=" * 60)

    common.save_json(common.result_path("dinov2_test.json"), {
        "dataset": common.DATASET,
        "num_classes": common.NUM_CLASSES,
        "backbone": backbone,
        "crop": crop,
        "test_accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "correct": correct,
        "total": int(len(y_test)),
        "class_names": common.class_names(),
    })
    print(f"Saved: {common.result_path('dinov2_test.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
