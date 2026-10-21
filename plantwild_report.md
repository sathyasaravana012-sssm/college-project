# PlantWild (34 classes) — training results

All four models trained on the PlantWild official split (5,398 train /
757 validation / 1,523 test), restricted to the 34 PlantVillage-compatible
classes. Test set = 1,523 images.

## Test-set comparison

| Model | PlantWild acc | PlantWild macro F1 | PlantVillage acc (38 cls) |
|---|---|---|---|
| Basic CNN | 10.05% | 8.44% | 93.98% |
| ResNet18 | 10.44% | 5.71% | 83.62% |
| MobileNetV2 | 38.67% | 35.69% | 90.36% |
| DINOv2 + logreg | 73.87% | 72.95% | 98.73% |

Notes:
- The PlantVillage column uses the same architectures on the 38-class
  dataset: the three TF models from `results/plantvillage_only.json`, the
  DINOv2 model from `dinov2_experiment/results/dinov2_plantvillage.json`
  (98.73% on 8,176 test images, earlier experiment).
- CNNs trained from scratch for 10 epochs on CPU; MobileNetV2 has an
  ImageNet-frozen backbone; DINOv2 is a frozen ViT-S/14 + logistic regression.

## Ensembles (test set, 1,523 images)

| Combination | acc | macro F1 | correct |
|---|---|---|---|
| equal average of all 4 | 62.31% | 60.91% | 949/1523 |
| validation-accuracy weighted ★ | 73.87% | 72.77% | 1125/1523 |
| DINOv2 + MobileNetV2 (0.5/0.5) | 71.63% | 70.63% | 1091/1523 |

★ Best ensemble. Validation-accuracy weights: Basic CNN 7%, ResNet18 7%, MobileNetV2 29%, DINOv2 + logreg 56%.

Equal averaging is *worse* than DINOv2 alone (62.31% vs 73.87%): two of the
four models are near chance (10%), so they pollute the mean. The weighted
average matches DINOv2 exactly on accuracy and is what the website uses,
while still showing every model's opinion side by side.

Live demo (Flask, single process, CPU): `venv\Scripts\python.exe website\app.py`
→ http://127.0.0.1:5000 (upload or pick a test sample; `/report` embeds the
confusion matrices and progress graphs).

## Per-class results — best model (DINOv2 + logreg)

| Class | support | precision | recall | F1 |
|---|---|---|---|---|
| cherry powdery mildew | 22 | 0.50 | 0.23 | 0.31 |
| tomato mosaic virus | 37 | 0.48 | 0.41 | 0.44 |
| tomato bacterial leaf spot | 56 | 0.47 | 0.45 | 0.46 |
| tomato septoria leaf spot | 44 | 0.57 | 0.39 | 0.46 |
| bell pepper leaf spot | 23 | 0.59 | 0.43 | 0.50 |
| corn gray leaf spot | 43 | 0.53 | 0.49 | 0.51 |
| tomato early blight | 69 | 0.46 | 0.59 | 0.52 |
| potato early blight | 45 | 0.60 | 0.47 | 0.53 |
| apple black rot | 34 | 0.73 | 0.47 | 0.57 |
| corn northern leaf blight | 44 | 0.58 | 0.64 | 0.61 |
| tomato late blight | 59 | 0.62 | 0.64 | 0.63 |
| tomato yellow leaf curl virus | 34 | 0.67 | 0.65 | 0.66 |
| potato late blight | 48 | 0.65 | 0.77 | 0.70 |
| apple scab | 58 | 0.66 | 0.78 | 0.71 |
| peach leaf | 31 | 0.83 | 0.65 | 0.73 |
| potato leaf | 49 | 0.69 | 0.86 | 0.76 |
| apple leaf | 88 | 0.77 | 0.78 | 0.78 |
| grape leaf spot | 21 | 0.80 | 0.76 | 0.78 |
| tomato leaf mold | 47 | 0.75 | 0.83 | 0.79 |
| tomato leaf | 45 | 0.77 | 0.82 | 0.80 |
| corn rust | 47 | 0.79 | 0.81 | 0.80 |
| grape leaf | 40 | 0.82 | 0.82 | 0.82 |
| apple rust | 61 | 0.81 | 0.85 | 0.83 |
| bell pepper leaf | 48 | 0.86 | 0.88 | 0.87 |
| cherry leaf | 57 | 0.88 | 0.86 | 0.87 |
| grape black rot | 45 | 0.89 | 0.89 | 0.89 |
| soybean leaf | 48 | 0.91 | 0.88 | 0.89 |
| squash powdery mildew | 56 | 0.88 | 0.95 | 0.91 |
| blueberry leaf | 56 | 0.93 | 0.93 | 0.93 |
| corn leaf | 31 | 0.97 | 0.90 | 0.93 |
| citrus greening disease | 47 | 0.90 | 0.98 | 0.94 |
| raspberry leaf | 36 | 0.92 | 0.97 | 0.95 |
| strawberry leaf | 39 | 1.00 | 0.92 | 0.96 |
| strawberry leaf scorch | 15 | 0.94 | 1.00 | 0.97 |

## Top confusions per model

### Basic CNN (10.05% acc)

- 18× `blueberry leaf` → `apple leaf`
- 17× `cherry leaf` → `apple leaf`
- 13× `potato leaf` → `apple leaf`
- 11× `tomato early blight` → `apple rust`
- 10× `raspberry leaf` → `apple leaf`
- 10× `grape leaf` → `apple leaf`
- 9× `tomato leaf` → `apple leaf`
- 9× `tomato late blight` → `apple rust`

### ResNet18 (10.44% acc)

- 36× `tomato early blight` → `grape black rot`
- 23× `potato early blight` → `grape black rot`
- 22× `apple leaf` → `grape black rot`
- 21× `cherry leaf` → `apple leaf`
- 20× `tomato leaf mold` → `bell pepper leaf`
- 19× `tomato bacterial leaf spot` → `grape black rot`
- 18× `tomato late blight` → `grape black rot`
- 18× `squash powdery mildew` → `bell pepper leaf`

### MobileNetV2 (38.67% acc)

- 15× `cherry leaf` → `apple leaf`
- 14× `corn gray leaf spot` → `corn northern leaf blight`
- 12× `tomato early blight` → `squash powdery mildew`
- 12× `potato leaf` → `squash powdery mildew`
- 12× `corn northern leaf blight` → `corn gray leaf spot`
- 9× `tomato mosaic virus` → `squash powdery mildew`
- 9× `tomato leaf` → `grape leaf`
- 9× `tomato early blight` → `tomato bacterial leaf spot`

### DINOv2 + logreg (73.87% acc)

- 18× `corn gray leaf spot` → `corn northern leaf blight`
- 16× `tomato bacterial leaf spot` → `tomato early blight`
- 11× `potato early blight` → `potato late blight`
- 10× `tomato septoria leaf spot` → `tomato bacterial leaf spot`
- 10× `corn northern leaf blight` → `corn gray leaf spot`
- 9× `tomato yellow leaf curl virus` → `tomato mosaic virus`
- 9× `tomato septoria leaf spot` → `tomato early blight`
- 9× `tomato late blight` → `tomato early blight`

## Hardest classes (DINOv2 F1 < 0.60)

- `cherry powdery mildew` — F1 0.31 (P 0.50 / R 0.23, n=22)
- `tomato mosaic virus` — F1 0.44 (P 0.48 / R 0.41, n=37)
- `tomato bacterial leaf spot` — F1 0.46 (P 0.47 / R 0.45, n=56)
- `tomato septoria leaf spot` — F1 0.46 (P 0.57 / R 0.39, n=44)
- `bell pepper leaf spot` — F1 0.50 (P 0.59 / R 0.43, n=23)
- `corn gray leaf spot` — F1 0.51 (P 0.53 / R 0.49, n=43)
- `tomato early blight` — F1 0.52 (P 0.46 / R 0.59, n=69)
- `potato early blight` — F1 0.53 (P 0.60 / R 0.47, n=45)
- `apple black rot` — F1 0.57 (P 0.73 / R 0.47, n=34)

Likely cause of the bulk of errors: within-crop disease pairs that differ
only by lesion pattern (early vs late blight, gray spot vs N. leaf blight,
mosaic vs yellow-leaf-curl virus) plus the ambiguous `X leaf` classes,
whose meaning in PlantWild (healthy? any disease?) is not documented.
