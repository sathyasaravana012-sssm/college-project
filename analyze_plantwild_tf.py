"""Per-class analysis of the three PlantWild TF models on the test set.

Loads each trained model, predicts on plantwild/split/test, computes
per-class precision/recall/F1 + confusion matrix, and saves:
    model/plantwild_tf_analysis.json

Run with the TF venv:  venv\\Scripts\\python.exe analyze_plantwild_tf.py
"""

import json
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

BASE = r"C:\Plant_disease_detection"
TEST_DIR = os.path.join(BASE, "plantwild", "split", "test")
MODELS = {
    "Basic CNN": os.path.join(BASE, "model", "plantwild_simple_cnn.keras"),
    "ResNet18": os.path.join(BASE, "model", "plantwild_resnet18.keras"),
    "MobileNetV2": os.path.join(BASE, "model", "plantwild_mobilenetv2.keras"),
}
# Basic CNN contains its own Rescaling(1/255) layer and expects raw 0-255
# input; ResNet18 / MobileNetV2 normalize externally and expect 0-1.
RAW_INPUT = {"Basic CNN"}
OUT_PATH = os.path.join(BASE, "model", "plantwild_tf_analysis.json")

ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=(224, 224), batch_size=32, shuffle=False, label_mode="int"
)
class_names = ds.class_names
ds = ds.prefetch(tf.data.AUTOTUNE)
ds_norm = ds.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

y_true = np.concatenate([y.numpy() for _, y in ds])
print(f"Test set: {len(y_true)} images, {len(class_names)} classes")

results = {}
for name, path in MODELS.items():
    print(f"\n=== {name} ===")
    model = tf.keras.models.load_model(path)
    model_ds = ds if name in RAW_INPUT else ds_norm
    y_prob = np.concatenate([model.predict(x, verbose=0) for x, _ in model_ds])
    y_pred = np.argmax(y_prob, axis=1)

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    per_class = [
        {
            "class_index": i,
            "class_name": class_names[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(s[i]),
        }
        for i in range(len(class_names))
    ]

    acc = float(np.mean(y_true == y_pred))
    print(f"  accuracy={acc*100:.2f}%")

    # worst classes
    worst = sorted(per_class, key=lambda c: c["f1"])[:5]
    for c in worst:
        print(f"  worst: {c['class_name']:<34} f1={c['f1']:.3f} "
              f"(P={c['precision']:.2f} R={c['recall']:.2f} n={c['support']})")

    # top confusion pairs (off-diagonal)
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                pairs.append((int(cm[i][j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    print("  top confusions:")
    for cnt, tr, pr in pairs[:5]:
        print(f"    {cnt:>4} x {tr} -> {pr}")

    results[name] = {
        "test_accuracy": acc,
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "correct": int(np.sum(y_true == y_pred)),
        "total": int(len(y_true)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": [
            {"count": c, "true": t, "predicted": pr} for c, t, pr in pairs[:20]
        ],
        "model_path": path,
    }

with open(OUT_PATH, "w", encoding="utf-8") as fh:
    json.dump({"dataset": "PlantWild (34 classes)", "split": "test",
               "class_names": class_names, "models": results}, fh, indent=2)
print(f"\nSaved: {OUT_PATH}")
