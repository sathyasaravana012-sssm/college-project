"""Collect the PlantWild test results of all four models into one table.

Reads:
  model/plantwild_simple_cnn_results.json
  model/plantwild_resnet18_results.json
  model/plantwild_mobilenetv2_results.json
  dinov2_experiment/results/plantwild_dinov2_test.json

Writes model/plantwild_summary.json and prints a markdown table.
Run with any Python (no TF / torch imports needed).
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

SOURCES = [
    ("Basic CNN",        BASE / "model" / "plantwild_simple_cnn_results.json"),
    ("ResNet18",         BASE / "model" / "plantwild_resnet18_results.json"),
    ("MobileNetV2",      BASE / "model" / "plantwild_mobilenetv2_results.json"),
    ("DINOv2 + logreg",  BASE / "dinov2_experiment" / "results" / "plantwild_dinov2_test.json"),
]

DINO_VAL = BASE / "dinov2_experiment" / "results" / "plantwild_dinov2_validation.json"

def load(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def pct(x):
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "-"

rows = []
missing = []
dino_val = load(DINO_VAL) or {}
for name, path in SOURCES:
    d = load(path)
    if d is None:
        missing.append(str(path))
        continue
    if name.startswith("DINOv2") and d.get("best_val_accuracy") is None:
        d["best_val_accuracy"] = dino_val.get("val_accuracy")
    rows.append({
        "model": name,
        "test_accuracy": d.get("test_accuracy"),
        "macro_precision": d.get("macro_precision"),
        "macro_recall": d.get("macro_recall"),
        "macro_f1": d.get("macro_f1"),
        "best_val_accuracy": d.get("best_val_accuracy"),
        "correct": d.get("correct"),
        "total": d.get("total"),
    })

if not rows:
    raise SystemExit("No result files found yet — nothing to collect.")

header = f"| {'Model':<18} | {'Test Acc':>9} | {'Macro P':>9} | {'Macro R':>9} | {'Macro F1':>9} | {'Correct':>8} |"
sep = "|" + "-" * 20 + "|" + "-" * 11 * 4 + "|" + "-" * 10 + "|"
print(f"\nPlantWild (34 classes) — test set results\n")
print(header)
print(sep)
for r in rows:
    correct = f"{r['correct']}/{r['total']}" if r["correct"] is not None else "-"
    print(f"| {r['model']:<18} | {pct(r['test_accuracy']):>9} | "
          f"{pct(r['macro_precision']):>9} | {pct(r['macro_recall']):>9} | "
          f"{pct(r['macro_f1']):>9} | {correct:>8} |")

if missing:
    print("\nStill missing (training not finished?):")
    for m in missing:
        print(" -", m)

out = BASE / "model" / "plantwild_summary.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump({"dataset": "PlantWild (34 classes)", "models": rows,
               "missing": missing}, f, indent=4)
print(f"\nSaved: {out}")
