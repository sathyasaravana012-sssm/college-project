"""Generate plantwild_report.md: PlantWild vs PlantVillage results,
per-class table (best model), and top confusions for every model.

Inputs:
  model/plantwild_tf_analysis.json           (3 TF models, per-class + confusion)
  dinov2_experiment/results/plantwild_dinov2_analysis.json
  results/plantvillage_only.json             (PlantVillage 38-class reference)

Run with any Python (json + pathlib only).
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

tf_an = load(BASE / "model" / "plantwild_tf_analysis.json")
dino_an = load(BASE / "dinov2_experiment" / "results" / "plantwild_dinov2_analysis.json")
pv = load(BASE / "results" / "plantvillage_only.json")
dino_pv = load(BASE / "dinov2_experiment" / "results" / "dinov2_plantvillage.json")

models_pw = {}
for name, r in tf_an["models"].items():
    models_pw[name] = r
models_pw["DINOv2 + logreg"] = dino_an

pv_results = pv["datasets"]["plantvillage_test"]["results"]
pv_lookup = {}
for name, r in pv_results.items():
    key = {"Simple CNN": "Basic CNN"}.get(name, name)
    pv_lookup[key] = r
pv_lookup["DINOv2 + logreg"] = dino_pv

lines = []
w = lines.append

w("# PlantWild (34 classes) — training results")
w("")
w("All four models trained on the PlantWild official split (5,398 train /")
w("757 validation / 1,523 test), restricted to the 34 PlantVillage-compatible")
w("classes. Test set = 1,523 images.")
w("")

# --- overall table ---
w("## Test-set comparison")
w("")
w("| Model | PlantWild acc | PlantWild macro F1 | PlantVillage acc (38 cls) |")
w("|---|---|---|---|")
order = ["Basic CNN", "ResNet18", "MobileNetV2", "DINOv2 + logreg"]
for name in order:
    r = models_pw[name]
    acc = r["test_accuracy"] * 100
    f1 = r["macro_f1"] * 100
    pv_acc = (f"{pv_lookup[name]['accuracy_percent']:.2f}%"
              if name in pv_lookup else "—")
    w(f"| {name} | {acc:.2f}% | {f1:.2f}% | {pv_acc} |")
w("")
w("Notes:")
w("- The PlantVillage column uses the same architectures on the 38-class")
w("  dataset: the three TF models from `results/plantvillage_only.json`, the")
w("  DINOv2 model from `dinov2_experiment/results/dinov2_plantvillage.json`")
w("  (98.73% on 8,176 test images, earlier experiment).")
w("- CNNs trained from scratch for 10 epochs on CPU; MobileNetV2 has an")
w("  ImageNet-frozen backbone; DINOv2 is a frozen ViT-S/14 + logistic regression.")
w("")

# --- ensembles ---
ens = load(BASE / "model" / "plantwild_ensemble_results.json")
best_ens = max(ens["ensembles"],
               key=lambda k: ens["ensembles"][k]["test_accuracy"])
w("## Ensembles (test set, 1,523 images)")
w("")
w("| Combination | acc | macro F1 | correct |")
w("|---|---|---|---|")
labels = {
    "mean_all": "equal average of all 4",
    "weighted_val": "validation-accuracy weighted",
    "dinov2_plus_mobilenet": "DINOv2 + MobileNetV2 (0.5/0.5)",
}
for name, m in ens["ensembles"].items():
    star = " ★" if name == best_ens else ""
    w(f"| {labels.get(name, name)}{star} | {m['test_accuracy']*100:.2f}% | "
      f"{m['macro_f1']*100:.2f}% | {m['correct']}/{m['total']} |")
w("")
w("★ Best ensemble. Validation-accuracy weights: "
  + ", ".join(f"{k} {v*100:.0f}%" for k, v in ens["weights"].items()) + ".")
w("")
w("Equal averaging is *worse* than DINOv2 alone (62.31% vs 73.87%): two of the")
w("four models are near chance (10%), so they pollute the mean. The weighted")
w("average matches DINOv2 exactly on accuracy and is what the website uses,")
w("while still showing every model's opinion side by side.")
w("")
w("Live demo (Flask, single process, CPU): `venv\\Scripts\\python.exe website\\app.py`")
w("→ http://127.0.0.1:5000 (upload or pick a test sample; `/report` embeds the")
w("confusion matrices and progress graphs).")
w("")

# --- per-class table for best model ---
w("## Per-class results — best model (DINOv2 + logreg)")
w("")
w("| Class | support | precision | recall | F1 |")
w("|---|---|---|---|---|")
for c in sorted(dino_an["per_class"], key=lambda c: c["f1"]):
    w(f"| {c['class_name']} | {c['support']} | {c['precision']:.2f} | "
      f"{c['recall']:.2f} | {c['f1']:.2f} |")
w("")

# --- top confusions per model ---
w("## Top confusions per model")
w("")
for name in order:
    r = models_pw[name]
    w(f"### {name} ({r['test_accuracy']*100:.2f}% acc)")
    w("")
    for tc in r["top_confusions"][:8]:
        w(f"- {tc['count']}× `{tc['true']}` → `{tc['predicted']}`")
    w("")

# --- hardest classes summary ---
w("## Hardest classes (DINOv2 F1 < 0.60)")
w("")
hard = [c for c in sorted(dino_an["per_class"], key=lambda c: c["f1"])
        if c["f1"] < 0.60]
for c in hard:
    w(f"- `{c['class_name']}` — F1 {c['f1']:.2f} "
      f"(P {c['precision']:.2f} / R {c['recall']:.2f}, n={c['support']})")
w("")
w("Likely cause of the bulk of errors: within-crop disease pairs that differ")
w("only by lesion pattern (early vs late blight, gray spot vs N. leaf blight,")
w("mosaic vs yellow-leaf-curl virus) plus the ambiguous `X leaf` classes,")
w("whose meaning in PlantWild (healthy? any disease?) is not documented.")
w("")

out = BASE / "plantwild_report.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {out}")
print(f"({len(lines)} lines)")
