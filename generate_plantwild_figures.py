"""Generate PlantWild figures for the website.

Outputs (website/static/figures/):
  confusion_basic_cnn.png
  confusion_resnet18.png
  confusion_mobilenetv2.png
  confusion_dinov2.png
  progress_models.png          (epoch-wise train/val curves, 3 TF models)
  progress_dinov2.png          (C sweep for the logreg head)

Run with the DINOv2 venv (has matplotlib):
  dinov2_experiment\\.venv\\Scripts\\python.exe generate_plantwild_figures.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np

plt = None  # imported lazily in main() so log parsing works without matplotlib


def _enable_pyplot():
    global plt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "website" / "static" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE / "logs"
LOG_PREFIX = "pw_"
LOG_STAMP = "20261018_204740"

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_log(path: Path) -> str:
    raw = path.read_bytes()
    candidates = [
        raw.decode("utf-16-le", errors="ignore").replace("\x00", ""),
        raw.decode("utf-8", errors="ignore").replace("\x00", ""),
    ]
    return max(candidates, key=lambda t: len(re.findall(r"Epoch \d+/\d+", t)))


def parse_history(log_path: Path):
    text = read_log(log_path)
    epochs = []
    cur = None
    for raw_line in text.splitlines():
        line = ANSI.sub("", raw_line)
        m = re.search(r"Epoch (\d+)/(\d+)", line)
        if m:
            if cur is not None:
                epochs.append(cur)
            cur = {"epoch": int(m.group(1)), "total": int(m.group(2)),
                   "train_acc": None, "train_loss": None,
                   "val_acc": None, "val_loss": None}
            continue
        if cur is None:
            continue
        m = re.search(r" - accuracy: ([0-9.]+) - loss: ([0-9.]+)", line)
        if m:
            cur["train_acc"] = float(m.group(1))
            cur["train_loss"] = float(m.group(2))
        m = re.search(r"val_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)", line)
        if m:
            cur["val_acc"] = float(m.group(1))
            cur["val_loss"] = float(m.group(2))
    if cur is not None:
        epochs.append(cur)
    return epochs


def confusion_figure(ax_free_plot, names, cm, title, out_path):
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ticks = np.arange(len(names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    _enable_pyplot()
    tf_an = json.loads((BASE / "model" / "plantwild_tf_analysis.json").read_text(encoding="utf-8"))
    dino_an = json.loads((BASE / "dinov2_experiment" / "results" / "plantwild_dinov2_analysis.json").read_text(encoding="utf-8"))
    names = tf_an["class_names"]

    figures = {
        "Basic CNN": "confusion_basic_cnn.png",
        "ResNet18": "confusion_resnet18.png",
        "MobileNetV2": "confusion_mobilenetv2.png",
    }
    for model, fname in figures.items():
        r = tf_an["models"][model]
        confusion_figure(None, names, r["confusion_matrix"],
                         f"{model} — PlantWild test confusion matrix "
                         f"(acc {r['test_accuracy']*100:.2f}%)",
                         FIG_DIR / fname)

    confusion_figure(None, names, dino_an["confusion_matrix"],
                     f"DINOv2 + logreg — PlantWild test confusion matrix "
                     f"(acc {dino_an['test_accuracy']*100:.2f}%)",
                     FIG_DIR / "confusion_dinov2.png")

    # --- training progress from the run logs ---
    logs = {
        "Basic CNN": LOG_DIR / f"{LOG_PREFIX}simple_cnn_{LOG_STAMP}.txt",
        "ResNet18": LOG_DIR / f"{LOG_PREFIX}resnet18_{LOG_STAMP}.txt",
        "MobileNetV2": LOG_DIR / f"{LOG_PREFIX}mobilenetv2_{LOG_STAMP}.txt",
    }
    histories = {}
    for model, path in logs.items():
        hist = parse_history(path)
        hist = [h for h in hist if h["train_acc"] is not None]
        histories[model] = hist
        print(f"{model}: {len(hist)} epochs, best val {max((h['val_acc'] or 0) for h in hist)*100:.2f}%")

    colors = {"Basic CNN": "#d62728", "ResNet18": "#1f77b4", "MobileNetV2": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    acc_ax, valacc_ax = axes[0]
    loss_ax, valloss_ax = axes[1]

    for model, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        acc_ax.plot(ep, [h["train_acc"] for h in hist], marker="o", ms=3,
                    color=colors[model], label=model)
        va = [(h["epoch"], h["val_acc"]) for h in hist if h["val_acc"] is not None]
        if va:
            valacc_ax.plot([e for e, _ in va], [v for _, v in va], marker="o",
                           ms=3, color=colors[model], label=model)
        loss_ax.plot(ep, [h["train_loss"] for h in hist], marker="o", ms=3,
                     color=colors[model], label=model)
        vl = [(h["epoch"], h["val_loss"]) for h in hist if h["val_loss"] is not None]
        if vl:
            valloss_ax.plot([e for e, _ in vl], [v for _, v in vl], marker="o",
                            ms=3, color=colors[model], label=model)

    dino_val = 0.7344782034346103
    valacc_ax.axhline(dino_val, color="#9467bd", ls="--", lw=1.5,
                      label=f"DINOv2 val acc ({dino_val*100:.1f}%)")
    valacc_ax.set_title("Validation accuracy (DINOv2 head is a single fit, no epochs)")

    acc_ax.set_title("Training accuracy")
    loss_ax.set_title("Training loss")
    valloss_ax.set_title("Validation loss")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in (acc_ax, valacc_ax):
        ax.set_ylabel("Accuracy")
    for ax in (loss_ax, valloss_ax):
        ax.set_ylabel("Loss")
        ax.set_yscale("log")

    fig.suptitle("PlantWild training progress (10 epochs, CPU)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "progress_models.png", dpi=140)
    plt.close(fig)

    # --- DINOv2 head sweep (the closest thing to epochs for a logreg) ---
    val = json.loads((BASE / "dinov2_experiment" / "results" /
                      "plantwild_dinov2_validation.json").read_text(encoding="utf-8"))
    cs = [c["C"] for c in val["candidates"]]
    accs = [c["val_accuracy"] * 100 for c in val["candidates"]]
    f1s = [c["val_macro_f1"] * 100 for c in val["candidates"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(cs))
    ax.plot(x, accs, marker="o", label="Validation accuracy")
    ax.plot(x, f1s, marker="s", label="Validation macro F1")
    best = int(np.argmax(accs))
    ax.scatter([x[best]], [accs[best]], s=140, facecolors="none",
               edgecolors="red", zorder=5, label=f"Selected C={cs[best]}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("LogisticRegression C")
    ax.set_ylabel("Percent")
    ax.set_title("DINOv2 head tuning (no epoch training — single fit per C)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "progress_dinov2.png", dpi=140)
    plt.close(fig)

    print(f"\nFigures written to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
