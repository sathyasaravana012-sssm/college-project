"""Per-class analysis of the PlantWild DINOv2 + logreg classifier on the TEST split.

Uses the cached test features and the fitted pipeline. Writes:
    results/plantwild_dinov2_analysis.json

    set PDD_DATASET=plantwild
    .venv\\Scripts\\python.exe analyze_test.py
"""

from __future__ import annotations

import sys

import numpy as np

import dinov2_common as common


def main(argv=None) -> int:
    import joblib
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    if common.DATASET != "plantwild":
        raise SystemExit("Run with PDD_DATASET=plantwild")

    paths = common.feature_cache_paths("test", common.BACKBONE_NAME, common.DINOV2_CROP)
    X = np.load(paths["features"]).astype(np.float32)
    y = np.load(paths["labels"]).astype(np.int64)
    names = common.class_names()
    n = len(names)

    model_path = common.artifact_path("dinov2_logreg.joblib")
    print(f"Loading: {model_path}")
    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X)

    p, r, f1, s = precision_recall_fscore_support(
        y, y_pred, labels=range(n), zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=range(n))

    per_class = [
        {
            "class_index": i,
            "class_name": names[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        for i in range(n)
    ]
    acc = float(np.mean(y == y_pred))

    print(f"accuracy={acc*100:.2f}%  macro_f1={np.mean(f1):.4f}")
    print("\nworst classes:")
    for c in sorted(per_class, key=lambda c: c["f1"])[:8]:
        print(f"  {c['class_name']:<34} f1={c['f1']:.3f} "
              f"(P={c['precision']:.2f} R={c['recall']:.2f} n={c['support']})")

    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((int(cm[i][j]), names[i], names[j]))
    pairs.sort(reverse=True)
    print("\ntop confusions:")
    for cnt, tr, pr in pairs[:10]:
        print(f"  {cnt:>4} x {tr} -> {pr}")

    common.save_json(common.result_path("dinov2_analysis.json"), {
        "dataset": common.DATASET,
        "split": "test",
        "class_names": names,
        "test_accuracy": acc,
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "correct": int(np.sum(y == y_pred)),
        "total": int(len(y)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": [
            {"count": c, "true": t, "predicted": pr} for c, t, pr in pairs[:20]
        ],
    })
    print(f"\nSaved: {common.result_path('dinov2_analysis.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
