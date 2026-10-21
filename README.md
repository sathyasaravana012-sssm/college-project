# Plant Disease Detection

This is a plant leaf disease classification project built with TensorFlow/Keras. It trains four CNN models on the PlantVillage dataset (38 classes), evaluates generalization on the PlantDoc real-world dataset, and provides a Flask web UI for interactive predictions.

## Dataset layout

- split/train, split/validation, split/test — PlantVillage (38 classes). Class indices are assigned using Python's ordinal sorted() of folder names. This order matches all models and predict.py.
- `real_world_dataset/real_world_dataset/PlantDoc-Dataset-master/train — PlantDoc training images (28 of 38 classes mapped). Not used for baseline training; used for fine-tuning.
- `real_world_dataset/real_world_dataset/PlantDoc-Dataset-master/test — PlantDoc test set (236 images, 27 folders). Used as the held-out generalization benchmark.
- `real_world_dataset/real_world_dataset/PlantDoc-Dataset-master/plantdoc_preprocessed — CLAHE-preprocessed copy of the PlantDoc test images.

A note on ordering: image_dataset_from_directory uses case-sensitive ASCII sorted(os.listdir(...)). This differs from case-insensitive Explorer/Excel sorts. Do not reorder folder names.

## Models

Four baseline checkpoints (trained on PlantVillage only):

| Model | File | Notes |
|---|---|---|
| basic (MobileNetV2) | model/plant_disease_model.keras | MobileNetV2 backbone with in-graph augmentation. |
| mobilenetv2 | model/mobilenetv2_plant_disease.keras | MobileNetV2 backbone with in-graph augmentation. |
| resnet18 | model/resnet18_plant_disease.keras | ResNet18 from scratch, no dropout/augmentation. |
| resnet18_improved | model/resnet18_improved.keras | ResNet18 from scratch + data augmentation + dropout. |

Fine-tuned on a mixed class-balanced set (PlantVillage + PlantDoc train):

| Model | File | PlantDoc test (fine-tuned) | Baseline |
|---|---|---|---|
| basic_plantdoc | model/basic_plantdoc.keras | **28.39%** | 26.27% |
| mobilenetv2_plantdoc | model/mobilenetv2_plantdoc.keras | 27.54% | 25.85% |
| resnet18_plantdoc | model/resnet18_plantdoc.keras | 13.56% | 5.93% |
| resnet18_improved_plantdoc | model/resnet18_improved_plantdoc.keras | 20.76% | 8.90% |

The PlantDoc scores are modest because field photos (clutter, background) differ from PlantVillage's clean leaf-on-white images. The two ResNet18 models gained the most (+7.63 and +11.86 pts) as they were undertrained by the original validation split; the MobileNets were already stronger on PlantDoc and moved least (+2.12 and +1.69 pts).

## Final evaluation

`evaluate_final_models.py` produces the four reported rows. Metrics come from `metrics.py`, which computes precision/recall/F1 from a confusion matrix in numpy only — scikit-learn is not a project dependency.

```bash
venv\Scripts\python.exe evaluate_final_models.py
```

Results are written to `results/final_evaluations.json`.

PlantVillage test (`split/test`, 8176 images, all 38 classes):

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|
| Simple CNN | 93.98% | 94.07% | 91.67% | 92.49% |
| ResNet18 | 83.62% | 85.96% | 82.85% | 81.91% |
| MobileNetV2 | 90.36% | 90.07% | 87.59% | 88.18% |
| **Ensemble** | **95.06%** | **95.10%** | **94.26%** | **94.50%** |

PlantDoc test (236 images, 27 classes present — see below):

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|---|---|---|---|---|
| Simple CNN | 26.27% | 34.43% | 26.71% | 26.45% |
| ResNet18 | 8.90% | 18.35% | 9.05% | 9.23% |
| MobileNetV2 | 25.85% | 27.90% | 25.91% | 23.93% |
| Ensemble | 25.85% | 39.19% | 26.03% | 26.24% |

Ensemble is unweighted soft voting on the mean of the three softmax outputs. No weights are fitted, so nothing here needs a held-out split and the result is deterministic.

The ensemble gains +1.08 pts on PlantVillage over its best member but loses 0.42 pts on PlantDoc. Its macro *precision* is the highest of all four rows on both datasets (95.10% and 39.19%) while its recall is close to Simple CNN's, meaning it is more conservative: fewer confident mistakes, slightly fewer hits.

Two caveats on these rows:

`PLANTDOC_MAPPING` covers 28 PlantVillage classes, but `Tomato two spotted spider mites leaf` exists only in the PlantDoc **train** folder. The test split therefore contains 27 of the 28, and macro averages cover those 27. Averaging over all 28 would fold in a class with no test images. PlantDoc is also imbalanced (6–12 images per class), so macro scores are sensitive to the small classes; `metrics.py` also records weighted averages in the JSON.

"Simple CNN" (`plant_disease_model.keras`) has a MobileNetV2 backbone with augmentation in the data pipeline, so it and the MobileNetV2 row are the same architecture family. Their predictions largely agree, which is why the ensemble tracks their consensus.

The ResNet18 checkpoint is 128.4 MB and gitignored, so this row is not reproducible from the repository alone without retraining.

## Web app

Flask app in `app.py` with central-crop preprocessing to match training. Models are selectable at runtime. Disease information for all 38 classes lives in `disease_info/disease_info.json`.

Six of the eight checkpoints are selectable in the app: `basic`, `basic_plantdoc`, `mobilenetv2`, `mobilenetv2_plantdoc`, `resnet18_improved`, and `resnet18_improved_plantdoc`. The two plain ResNet18 checkpoints are not offered, because they scored lowest of the four (13.56% on PlantDoc, against 20.76% for the improved variant) and cost 128.4 MB each to load. Their weights remain in `model/` and `train_resnet18.py` / `evaluate_resnet18.py` still work, so the accuracy table above stays reproducible.

Images can be uploaded (drag & drop or file browse) or captured with a device camera using the "Take photo" button. Capture uses a dedicated `<input type="file" capture="environment">` with no `name` attribute, so it never posts on its own; `static/js/main.js` copies the captured file into the main `#image` input via `DataTransfer` and the form still submits a single `image` field. Unlike a live `getUserMedia` viewfinder, this requires no secure context, so it works over plain HTTP on a LAN address as well as on `127.0.0.1`. Camera photos are field images and therefore land in the ~21-28% real-world accuracy band rather than the PlantVillage validation figure.

Run the app:

```bash
venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000/ in your browser. Supported uploads: PNG, JPG, JPEG, WebP, BMP, GIF (max 16 MB).

## CLI prediction

predict.py loads a single model and predicts on `test_images/test5.jpg` by default. It uses the same central-crop preprocessing as the app.

```bash
venv\Scripts\python.exe predict.py
```

## Fine-tuning

`finetune_plantdoc.py` builds a class-balanced mixed dataset (PlantVillage + PlantDoc train), oversamples PlantDoc, and fine-tunes any of the four checkpoints. Validation is on PlantVillage only; the PlantDoc test set is never used for training or selection.

```bash
venv\Scripts\python.exe finetune_plantdoc.py --model mobilenetv2 --epochs 5
```

## PlantWild (34 classes)

`plantwild/` holds the PlantWild field-photograph dataset restricted to the 34 PlantVillage-compatible classes (5,398 train / 757 validation / 1,523 test). Only the small metadata files are tracked; the images are regenerable from `plantwild/readme.txt`. Three models were trained from scratch on CPU via `run_plantwild_training.ps1` (`train_plantwild_simple_cnn.py`, `train_plantwild_resnet18.py`, `train_plantwild_mobilenetv2.py`) and evaluated by `evaluate_final_ensemble.py` / `ensemble_plantwild.py`:

| Model | PlantWild test acc | macro F1 |
|---|---|---|
| Basic CNN | 10.05% | 8.44% |
| ResNet18 | 10.44% | 5.71% |
| MobileNetV2 | 38.67% | 35.69% |
| DINOv2 + logreg | 73.87% | 72.95% |
| **Weighted ensemble** | **73.87%** | 72.77% |

The ensemble is validation-accuracy weighted soft voting (`model/plantwild_ensemble_results.json`); equal averaging was measurably worse (62.31%) because two members sit near chance. Per-class scores, hardest classes, and top confusions are in `plantwild_report.md` (regenerate with `make_plantwild_report.py` and `generate_plantwild_figures.py`).

## DINOv2 experiment

`dinov2_experiment/` is a self-contained PyTorch experiment (own venv, see `dinov2_experiment/README.md`) that freezes `dinov2_vits14` features and fits a logistic-regression head:

| Dataset | DINOv2 + logreg | MobileNetV2 |
|---|---|---|
| PlantVillage test (38 classes) | 98.73% | 94.32% |
| PlantDoc test (27 classes, real world) | **41.95%** | 24.58% |

41.95% is the best real-world number in the project (against 28.39% for the best TF model) but rests on only 236 images across 27 classes — treat it as directional, not conclusive.

## PlantWild website

A second Flask app, `website/app.py`, serves the four-model PlantWild weighted ensemble with a `/report` page (confusion matrices and training graphs):

```bash
venv\Scripts\python.exe website\app.py
```

Open http://127.0.0.1:5000 — upload a leaf photo or pick a test sample. Smoke test: `venv\Scripts\python.exe website\test_app.py`.

## Environment

Install dependencies:

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Note: TensorFlow GPU support is limited on native Windows for TF 2.11+. This code runs on CPU.

## Caveats

- split/validation had two misnamed folders (P_Eb, Tomato_Early_Blight) that were corrected in this session. All 38 class indices now align across splits.
- The PlantDoc mapping covers 28 of 38 PlantVillage classes. Classes without PlantDoc coverage still produce predictions but have no real-world validation examples.
- Absolute PlantDoc accuracy is low (~21–28%). This is primarily a domain gap (field vs lab images), not a preprocessing bug; central-crop preprocessing matches the training pipeline and gives 28.39% for the best model.
- Six checkpoints exceed GitHub's 100 MB per-file limit and are gitignored: the four ResNet18 variants plus `plantwild_resnet18.keras` (128 MB), `plantwild_simple_cnn.keras` (148 MB), and `simple_cnn_plant_disease.keras` (148 MB). Rows that depend on them (see `plantwild_report.md`, `results/final_evaluations.json`) are not reproducible from a fresh clone without retraining.
