import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PLANTVILLAGE_TRAIN_DIR = BASE_DIR / "split" / "train"
PLANTVILLAGE_VAL_DIR = BASE_DIR / "split" / "validation"
PLANTVILLAGE_TEST_DIR = BASE_DIR / "split" / "test"

PLANTDOC_ROOT = (
    BASE_DIR / "real_world_dataset" / "real_world_dataset" / "Plantdoc-Dataset-master"
)
PLANTDOC_TRAIN_DIR = PLANTDOC_ROOT / "train"
PLANTDOC_TEST_DIR = PLANTDOC_ROOT / "test"
PLANTDOC_PREPROCESSED_DIR = PLANTDOC_ROOT / "plantdoc_preprocessed"

MODEL_DIR = BASE_DIR / "model"
DISEASE_INFO_PATH = BASE_DIR / "disease_info" / "disease_info.json"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# Model checkpoints, ordered from weakest/earliest to strongest. These are the
# models the app offers and finetune_plantdoc.py accepts. The plain ResNet18
# checkpoints (resnet18_plant_disease.keras, resnet18_plantdoc.keras) are
# deliberately absent: they scored lowest of the four on PlantDoc (13.56%
# fine-tuned vs 20.76% for the improved variant) and cost 128.4 MB each to load.
# The weights and their training/evaluation scripts are still in the repo, so
# train_resnet18.py and evaluate_resnet18.py continue to work.
BASELINE_MODELS = {
    "basic": MODEL_DIR / "plant_disease_model.keras",
    "basic_plantdoc": MODEL_DIR / "basic_plantdoc.keras",
    "resnet18_improved": MODEL_DIR / "resnet18_improved.keras",
    "resnet18_improved_plantdoc": MODEL_DIR / "resnet18_improved_plantdoc.keras",
    "mobilenetv2": MODEL_DIR / "mobilenetv2_plant_disease.keras",
    "mobilenetv2_plantdoc": MODEL_DIR / "mobilenetv2_plantdoc.keras",
}

MODELS_WITH_GRAPH_AUGMENTATION = {
    "basic",
    "mobilenetv2",
}


def _derive_class_names():
    if not PLANTVILLAGE_TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Training folder not found: {PLANTVILLAGE_TRAIN_DIR}")

    names = sorted(
        folder
        for folder in os.listdir(PLANTVILLAGE_TRAIN_DIR)
        if (PLANTVILLAGE_TRAIN_DIR / folder).is_dir()
    )

    if len(names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, but found {len(names)}")

    return names


# Derived from disk rather than hardcoded. image_dataset_from_directory assigns
# integer labels using sorted(os.listdir(...)), which is ordinal code-point order
# -- note that 'P' < 'h', 'T' < 'h' and 'Y' < 'm'. A case-insensitive sort
# (Excel, Windows Explorer) produces a different order and silently mislabels.
CLASS_NAMES = _derive_class_names()

CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# PlantDoc folder name -> PlantVillage class name. Covers 28 of 38 classes;
# the remaining 10 have no PlantDoc equivalent (Apple black rot, Corn healthy,
# Grape esca, Grape leaf blight, Orange haunglongbing, Peach bacterial spot,
# Potato healthy, Strawberry leaf scorch, Tomato target spot, Cherry mildew).
PLANTDOC_MAPPING = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf":
        "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

_UNMAPPED = sorted(
    target for target in PLANTDOC_MAPPING.values() if target not in CLASS_TO_INDEX
)
if _UNMAPPED:
    raise ValueError(f"PLANTDOC_MAPPING targets not in class list: {_UNMAPPED}")

PLANTDOC_TO_INDEX = {
    folder: CLASS_TO_INDEX[target]
    for folder, target in PLANTDOC_MAPPING.items()
}


def class_folders(directory):
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        folder
        for folder in os.listdir(directory)
        if (Path(directory) / folder).is_dir()
    )


def image_files(directory):
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        entry
        for entry in os.listdir(directory)
        if (directory / entry).is_file()
        and Path(entry).suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_split_alignment():
    train = class_folders(PLANTVILLAGE_TRAIN_DIR)
    validation = class_folders(PLANTVILLAGE_VAL_DIR)
    test = class_folders(PLANTVILLAGE_TEST_DIR)

    if train != validation:
        mismatches = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(train, validation))
            if a != b
        ]
        raise ValueError(
            "split/validation class order does not match split/train.\n"
            "Integer labels are assigned per-dataset, so this shifts every\n"
            "validation label and corrupts val_accuracy / val_loss.\n"
            f"Mismatched indices: {mismatches}"
        )

    if train != test:
        raise ValueError("split/test class order does not match split/train")

    return train


def count_images(directory, recursive=True):
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"

    return sum(
        1
        for entry in directory.glob(pattern)
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    )