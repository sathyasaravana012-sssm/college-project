"""Train a lightweight logistic-regression classifier on frozen DINOv2 features.

Uses ONLY the training features to fit. The validation features are used solely
to pick the regularization strength ``C``. The PlantVillage test split is never
touched here.

    .venv\\Scripts\\python.exe train_classifier.py
    .venv\\Scripts\\python.exe train_classifier.py --C 0.1 1 10 --max-iter 2000

Outputs
-------
models/dinov2_logreg.joblib   fitted pipeline (scaler + logistic regression)
models/class_mapping.json     class-index -> class-name mapping + metadata
results/dinov2_validation.json validation metrics for every C tried
"""

from __future__ import annotations

import argparse
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--C", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0],
                        help="Regularization strengths to try (validation-selected).")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--no-standardize", action="store_true",
                        help="Skip feature standardization.")
    parser.add_argument("--backbone", default=common.BACKBONE_NAME)
    parser.add_argument("--crop", type=int, default=common.DINOV2_CROP)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    print("Loading cached features...")
    X_train, y_train = _load_split("train", args.backbone, args.crop)
    X_val, y_val = _load_split("validation", args.backbone, args.crop)
    print(f"  train: {X_train.shape}, val: {X_val.shape}")

    results = []
    best = {"macro_f1": -1.0, "C": None, "pipeline": None, "val_accuracy": 0.0}

    for C in args.C:
        steps = []
        if not args.no_standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LogisticRegression(
            C=C, max_iter=args.max_iter, solver="lbfgs",
        )))
        pipeline = Pipeline(steps)

        print(f"\nFitting LogisticRegression C={C} (max_iter={args.max_iter})...")
        start = time.time()
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        elapsed = time.time() - start

        acc = float(accuracy_score(y_val, val_pred))
        macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
        print(f"  val accuracy={acc*100:.3f}%  macro_f1={macro_f1:.4f}  ({elapsed:.1f}s)")

        results.append({"C": C, "val_accuracy": acc, "val_macro_f1": macro_f1,
                        "seconds": elapsed})

        if macro_f1 > best["macro_f1"]:
            best.update({"macro_f1": macro_f1, "C": C,
                         "pipeline": pipeline, "val_accuracy": acc})

    if best["pipeline"] is None:
        raise SystemExit("No model was trained; check the --C values.")

    print("\n" + "=" * 60)
    print(f"Best C = {best['C']} (val macro_f1={best['macro_f1']:.4f}, "
          f"val accuracy={best['val_accuracy']*100:.3f}%)")
    print("=" * 60)

    model_path = common.artifact_path("dinov2_logreg.joblib")
    import joblib
    joblib.dump(best["pipeline"], model_path)

    common.save_json(common.artifact_path("class_mapping.json"), {
        "class_names": common.class_names(),
        "num_classes": common.NUM_CLASSES,
        "dataset": common.DATASET,
        "backbone": args.backbone,
        "feature_dim": common.FEATURE_DIM,
        "crop": args.crop,
        "resize": common.DINOV2_RESIZE,
        "mean": common.IMAGENET_MEAN,
        "std": common.IMAGENET_STD,
        "selected_C": best["C"],
        "val_macro_f1": best["macro_f1"],
        "val_accuracy": best["val_accuracy"],
    })
    common.save_json(common.result_path("dinov2_validation.json"), {
        "backbone": args.backbone,
        "candidates": results,
        "selected_C": best["C"],
        "val_accuracy": best["val_accuracy"],
        "val_macro_f1": best["macro_f1"],
        "train_size": int(len(y_train)),
        "val_size": int(len(y_val)),
    })

    print(f"Saved classifier : {model_path}")
    print(f"Saved class map  : {common.artifact_path('class_mapping.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
