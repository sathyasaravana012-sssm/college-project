# DINOv2 experiment for Plant Disease Detection

A self-contained, reproducible experiment that tests whether **frozen DINOv2
features + a lightweight logistic-regression classifier** classify leaf images
better than the existing **MobileNetV2** baseline.

It does **not** modify, retrain, or overwrite the existing MobileNetV2 model,
the dataset splits, the Flask application, or any existing evaluation results.
Everything new lives under `dinov2_experiment\`.

- Backbone: `dinov2_vits14` (ViT-S/14, 384-d CLS features), **frozen**
- Classifier: scikit-learn `LogisticRegression`
- Preprocessing: DINOv2's own eval transform (`Resize(256) -> CenterCrop(224)
  -> ImageNet normalize`) - **not** the MobileNetV2 `[0, 1]` scaling
- Classes: the same 38 PlantVillage classes, same index order as the existing
  pipeline (verified against `config.CLASS_NAMES`)

> **Path note.** The live project root is `C:\Plant_disease_detection`
> (underscore) and the splits live in `split\{train,validation,test}` (not
> `data_split`). All paths in this experiment are resolved relative to this
> folder, so nothing is hard-coded.

---

## 1. Why an isolated environment

The project's main venv has TensorFlow; this experiment needs PyTorch
(`torch-2.10.0+cpu-cp313`, compatible with your Python 3.13). To avoid touching
the TensorFlow environment, install PyTorch into a separate venv here.

First run needs internet access **once**: `torch.hub` downloads the DINOv2 repo
from GitHub and the pretrained weights from `dl.fbaipublicfiles.com`. After that
everything is cached and works offline.

## 2. Setup (Windows PowerShell)

From `C:\Plant_disease_detection`:

```powershell
cd dinov2_experiment
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Verify the environment

```powershell
.\.venv\Scripts\python.exe check_dinov2.py
```

Expected: versions printed, backbone loaded, `Feature shape : (2, 384)`,
`RESULT: OK - backbone works.` and `found 38 classes`.

---

## 4. Run the experiment (in order)

### Step A - extract features (the slow step)

```powershell
# Quick smoke test first: 300 images/class of the train split (~a few minutes)
.\.venv\Scripts\python.exe extract_features.py --splits train --limit-per-class 300

# Then the full extraction (train + validation + test)
.\.venv\Scripts\python.exe extract_features.py --batch-size 32
```

- `--batch-size` is configurable (16-64 is reasonable on CPU).
- Features are cached in `features\`. Interrupting and re-running **resumes**.
- Expected workload: ~37,982 train + ~8,126 val + ~8,176 test images. On CPU,
  ViT-S/14 is roughly a few images per second to ~20 img/s depending on your
  CPU, so the full train split is **likely 1-4 hours**. Validation/test are much
  faster. Run it when you can leave the PC on. The smoke test proves the
  pipeline before committing to the long run.

### Step B - train the classifier

```powershell
.\.venv\Scripts\python.exe train_classifier.py
```

Fits LogisticRegression on **training features only**; picks `C` using the
**validation** features. Writes `models\dinov2_logreg.joblib` and
`models\class_mapping.json`. Test images are never used here.

### Step C - collect MobileNetV2 predictions (in the TensorFlow venv)

The DINOv2 venv has no TensorFlow, so measure the baseline with the project's
own interpreter:

```powershell
# still from C:\Plant_disease_detection
.\venv\Scripts\python.exe dinov2_experiment\collect_mobilenetv2_predictions.py
```

This reads only images and writes `dinov2_experiment\results\mobilenetv2_*_preds.npz`.
`--model-key basic` (default) is `plant_disease_model.keras`, the MobileNetV2
checkpoint with 93.98% PlantVillage test accuracy.

### Step D - evaluate and compare

```powershell
.\.venv\Scripts\python.exe evaluate_models.py --dataset both
```

Scores both models on **exactly the same images** with the project's own
`metrics.py`, and writes under `results\`: accuracy, macro/weighted
precision-recall-F1, per-class tables (`per_class_*.csv`), confusion matrices
(`confusion_*.csv/.png`), and misclassified examples (`misclassified_*.json`).

### Step E - predict a single image

```powershell
.\.venv\Scripts\python.exe predict_leaf.py "C:\path\to\leaf.jpg"
.\.venv\Scripts\python.exe predict_leaf.py leaf.jpg --top-k 5
```

---

## 5. Real-world evaluation

The real-world photographs are the **PlantDoc test split**
(`real_world_dataset\...\PlantDoc-Dataset-master\test`, 236 images, 27 mapped
classes). It is never used for training or tuning.

`evaluate_models.py` reports, for both models: accuracy, macro F1, per-class
recall, and a confusion matrix for the classes actually present, plus a list of
misclassified files for manual inspection.

**Sample size warning:** 236 images across 27 classes (often <10 per class) is
far too small for reliable per-class conclusions. Treat these numbers as
directional only. Do not claim DINOv2 improves real-world performance unless
the measured numbers demonstrate it. If you later fine-tune on real-world data,
make that a separate experiment with its own held-out test set.

## 6. Files

| File | Purpose |
| --- | --- |
| `dinov2_common.py` | paths, class mapping, preprocessing, frozen backbone loader |
| `check_dinov2.py` | environment + model-loading verification |
| `extract_features.py` | cache train/val/test/real-world features (resumable) |
| `train_classifier.py` | train + validation-select the logistic-regression classifier |
| `collect_mobilenetv2_predictions.py` | run MobileNetV2 in the TF venv and dump predictions |
| `evaluate_models.py` | score both models on identical images and compare |
| `predict_leaf.py` | single-image DINOv2 prediction (top-k probabilities) |

Outputs: `features\` (cached arrays), `models\` (classifier + class map),
`results\` (metrics, confusion matrices, comparisons, misclassified lists).

## 7. Optional: integrate into the Flask app later

Do **not** replace the current model. After evaluation, a safe integration is a
selectable model option: add a `dinov2` entry to the app's model list that
routes to `predict_leaf.py`'s pipeline (torch) while MobileNetV2 keeps using its
existing TensorFlow path. That requires running both torch and TensorFlow in the
same process/environment, or shelling out to the DINOv2 venv. Keep the current
MobileNetV2 path as the default until the real-world numbers justify a switch.
