# Plant Disease Detection

This is a plant leaf disease classification project built with TensorFlow/Keras. It trains four CNN models on the PlantVillage dataset (38 classes), evaluates generalization on the PlantDoc real-world dataset, and provides a Flask web UI for interactive predictions.

## Dataset layout

- split/train, split/validation, split/test — PlantVillage (38 classes). Class indices are assigned using Python's ordinal sorted() of folder names. This order matches all models and predict.py.
- eal_world_dataset/real_world_dataset/PlantDoc-Dataset-master/train — PlantDoc training images (28 of 38 classes mapped). Not used for baseline training; used for fine-tuning.
- eal_world_dataset/real_world_dataset/PlantDoc-Dataset-master/test — PlantDoc test set (236 images, 27 folders). Used as the held-out generalization benchmark.
- eal_world_dataset/real_world_dataset/PlantDoc-Dataset-master/plantdoc_preprocessed — CLAHE-preprocessed copy of the PlantDoc test images.

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

## Web app

Flask app in pp.py with central-crop preprocessing to match training. Models are selectable at runtime. Disease information for all 38 classes lives in disease_info/disease_info.json.

Run the app:

`ash
venv\Scripts\python.exe app.py
`

Open http://127.0.0.1:5000/ in your browser. Supported uploads: PNG, JPG, JPEG, WebP, BMP, GIF (max 16 MB).

## CLI prediction

predict.py loads a single model and predicts on 	est_images/test5.jpg by default. It uses the same central-crop preprocessing as the app.

`ash
venv\Scripts\python.exe predict.py
`

## Fine-tuning

inetune_plantdoc.py builds a class-balanced mixed dataset (PlantVillage + PlantDoc train), oversamples PlantDoc, and fine-tunes any of the four checkpoints. Validation is on PlantVillage only; the PlantDoc test set is never used for training or selection.

`ash
venv\Scripts\python.exe finetune_plantdoc.py --model mobilenetv2 --epochs 5
`

## Environment

Install dependencies:

`ash
venv\Scripts\python.exe -m pip install -r requirements.txt
`

Note: TensorFlow GPU support is limited on native Windows for TF 2.11+. This code runs on CPU.

## Caveats

- split/validation had two misnamed folders (P_Eb, Tomato_Early_Blight) that were corrected in this session. All 38 class indices now align across splits.
- The PlantDoc mapping covers 28 of 38 PlantVillage classes. Classes without PlantDoc coverage still produce predictions but have no real-world validation examples.
- Absolute PlantDoc accuracy is low (~21–28%). This is primarily a domain gap (field vs lab images), not a preprocessing bug; central-crop preprocessing matches the training pipeline and gives 28.39% for the best model.
