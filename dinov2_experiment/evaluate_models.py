"""Evaluate and compare MobileNetV2 and DINOv2 on identical images.

Runs in the isolated DINOv2 environment. DINOv2 predictions come from the
frozen backbone + the logistic-regression classifier. MobileNetV2 predictions
are read from files produced by ``collect_mobilenetv2_predictions.py`` (run in
the TensorFlow venv). Both models are scored on exactly the same images with
the same ground-truth labels and the same 38-class order.

    .venv\\Scripts\\python.exe evaluate_models.py --dataset both --model-key basic

Metrics come from the project's own ``metrics.py`` (numpy-only confusion matrix
+ macro/weighted averages) so the numbers match the rest of the project.

Outputs (under dinov2_experiment/results):
    dinov2_plantvillage.json / dinov2_realworld.json
    mobilenetv2_plantvillage.json / mobilenetv2_realworld.json
    comparison_summary.json
    confusion_<model>_<dataset>.csv / .png
    per_class_<model>_<dataset>.csv
    misclassified_<model>_<dataset>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dinov2_common as common  # noqa: E402
import metrics  # noqa: E402  (project-root numpy metrics)


# ---------------------------------------------------------------------------
# DINOv2 predictions
# ---------------------------------------------------------------------------

def _load_classifier():
    import joblib

    model_path = common.MODELS_DIR / "dinov2_logreg.joblib"
    if not model_path.exists():
        raise SystemExit("No trained classifier. Run train_classifier.py first.")
    return joblib.load(model_path)


def dinov2_predictions(dataset: str, args):
    pipeline = _load_classifier()

    if dataset == "plantvillage":
        paths = common.feature_cache_paths("test", args.backbone, args.crop)
        if not paths["features"].exists():
            raise SystemExit("Missing cached test features. Run extract_features.py --splits test")
        X = np.load(paths["features"])
        y_true = np.load(paths["labels"])
        image_paths = common.load_json(paths["paths"])["paths"]
    elif dataset == "realworld":
        paths = common.feature_cache_paths("realworld", args.backbone, args.crop)
        if paths["features"].exists():
            X = np.load(paths["features"])
            y_true = np.load(paths["labels"])
            image_paths = common.load_json(paths["paths"])["paths"]
        else:
            records = common.plantdoc_records(common.REAL_WORLD_DIR)
            X, y_true, image_paths = _extract_live(records, args)
    else:
        raise SystemExit(f"Unknown dataset: {dataset}")

    probs = pipeline.predict_proba(X)
    y_pred = probs.argmax(axis=1)
    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_pred": np.asarray(y_pred, dtype=np.int64),
        "probs": np.asarray(probs, dtype=np.float32),
        "paths": list(image_paths),
    }


def _extract_live(records, args):
    from PIL import Image
    import torch

    backbone = common.load_backbone(device=args.device)
    features, labels, paths = [], [], []
    for i, (path, label) in enumerate(records):
        try:
            with Image.open(path) as img:
                tensor = common.preprocess_pil(img)
            feats = common.extract_features_batch(
                backbone, tensor.unsqueeze(0), args.device)
            features.append(feats[0])
            labels.append(label)
            paths.append(path)
        except Exception as exc:
            print(f"[warn] could not read {path}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(records)}")
    return np.asarray(features, dtype=np.float32), labels, paths


# ---------------------------------------------------------------------------
# MobileNetV2 predictions
# ---------------------------------------------------------------------------

def mobilenet_predictions(dataset: str):
    npz_path = common.RESULTS_DIR / f"mobilenetv2_{dataset}_preds.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "y_true": data["y_true"].astype(np.int64),
        "y_pred": data["y_pred"].astype(np.int64),
        "probs": data["probs"].astype(np.float32) if "probs" in data else None,
        "paths": list(data["paths"]) if "paths" in data else None,
        "model_key": str(data["model_key"]) if "model_key" in data else "mobilenetv2",
    }


# ---------------------------------------------------------------------------
# Scoring + saving
# ---------------------------------------------------------------------------

def _save_confusion_csv(matrix, names, path: Path):
    header = "true\\pred," + ",".join(names)
    lines = [header]
    for i, row in enumerate(matrix):
        lines.append(names[i] + "," + ",".join(str(int(v)) for v in row))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _save_confusion_png(matrix, names, path: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[note] matplotlib unavailable ({exc}); skipping PNG {path.name}")
        return
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[saved] {path}")


def evaluate(name: str, dataset: str, preds, class_names):
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]

    present = sorted(set(int(v) for v in y_true))
    summary = metrics.summarize(y_true, y_pred, common.NUM_CLASSES,
                                class_names=class_names, class_indices=present)
    summary["model"] = name
    summary["dataset"] = dataset
    summary["images"] = int(len(y_true))
    summary["classes_present"] = len(present)

    stem = f"{name}_{dataset}"
    common.save_json(common.RESULTS_DIR / f"{stem}.json", summary)

    matrix = np.asarray(summary["confusion_matrix"])
    _save_confusion_csv(matrix, class_names,
                        common.RESULTS_DIR / f"confusion_{stem}.csv")
    _save_confusion_png(matrix, class_names,
                        common.RESULTS_DIR / f"confusion_{stem}.png",
                        f"{name} on {dataset}")

    per_class_lines = ["class_index,class_name,precision,recall,f1,support"]
    for row in summary["per_class"]:
        per_class_lines.append(
            f"{row['class_index']},{row['class_name']},"
            f"{row['precision']:.6f},{row['recall']:.6f},{row['f1']:.6f},{row['support']}"
        )
    (common.RESULTS_DIR / f"per_class_{stem}.csv").write_text(
        "\n".join(per_class_lines), encoding="utf-8")

    # Misclassified examples for manual inspection.
    misclassified = []
    probs = preds.get("probs")
    paths = preds.get("paths")
    for i in range(len(y_true)):
        if int(y_true[i]) != int(y_pred[i]):
            entry = {
                "path": paths[i] if paths is not None else None,
                "true_index": int(y_true[i]),
                "true_class": class_names[int(y_true[i])],
                "pred_index": int(y_pred[i]),
                "pred_class": class_names[int(y_pred[i])],
            }
            if probs is not None:
                entry["pred_confidence"] = float(probs[i][int(y_pred[i])])
            misclassified.append(entry)
    common.save_json(common.RESULTS_DIR / f"misclassified_{stem}.json", {
        "count": len(misclassified),
        "note": "pred_confidence is the model's softmax/probability, not proof of correctness.",
        "examples": misclassified,
    })

    print(
        f"{name:<14}{dataset:<13} acc={summary['accuracy_percent']:6.2f}%  "
        f"macroP={summary['macro_precision']*100:6.2f}%  "
        f"macroR={summary['macro_recall']*100:6.2f}%  "
        f"macroF1={summary['macro_f1']*100:6.2f}%  "
        f"weightedF1={summary['weighted_f1']*100:6.2f}%  "
        f"({summary['correct']}/{summary['total']})"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="both",
                        choices=["plantvillage", "realworld", "both"])
    parser.add_argument("--model-key", default="basic",
                        help="Which MobileNetV2 prediction file to compare against.")
    parser.add_argument("--backbone", default=common.BACKBONE_NAME)
    parser.add_argument("--crop", type=int, default=common.DINOV2_CROP)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    class_names = common.class_names()

    datasets = ["plantvillage", "realworld"] if args.dataset == "both" else [args.dataset]
    all_summaries = []

    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    for dataset in datasets:
        print(f"\n--- dataset: {dataset} ---")

        dn = dinov2_predictions(dataset, args)
        all_summaries.append(evaluate("dinov2", dataset, dn, class_names))

        mn = mobilenet_predictions(dataset)
        if mn is None:
            print(f"[warn] no MobileNetV2 predictions for '{dataset}'.")
            print("       Run collect_mobilenetv2_predictions.py in the TensorFlow venv.")
        else:
            if len(mn["y_true"]) != len(dn["y_true"]):
                print(f"[warn] image count mismatch: MobileNetV2={len(mn['y_true'])} "
                      f"DINOv2={len(dn['y_true'])}; skipping direct comparison.")
            all_summaries.append(evaluate("mobilenetv2", dataset, mn, class_names))

    common.save_json(common.RESULTS_DIR / "comparison_summary.json", {
        "mobilenet_model_key": args.model_key,
        "dinov2_backbone": args.backbone,
        "summaries": [
            {
                "model": s["model"], "dataset": s["dataset"],
                "images": s["images"], "classes_present": s["classes_present"],
                "accuracy_percent": s["accuracy_percent"],
                "macro_precision": s["macro_precision"],
                "macro_recall": s["macro_recall"], "macro_f1": s["macro_f1"],
                "weighted_precision": s["weighted_precision"],
                "weighted_recall": s["weighted_recall"],
                "weighted_f1": s["weighted_f1"],
            }
            for s in all_summaries
        ],
    })

    print("\nSaved comparison summary:", common.RESULTS_DIR / "comparison_summary.json")
    print("Real-world note: report sample sizes. PlantDoc test has ~236 images over 27 "
          "classes - too small for confident per-class conclusions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
