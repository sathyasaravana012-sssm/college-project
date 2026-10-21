"""Shared helpers for the DINOv2 experiment.

Why a separate module and not the project's ``config.py``?
    The project's ``config.py`` is imported by the TensorFlow application. To
    keep the DINOv2 experiment runnable inside its OWN isolated virtual
    environment (torch + scikit-learn, no TensorFlow), this module re-implements
    just the small pieces it needs: dataset discovery and the exact class
    ordering. Nothing here imports TensorFlow.

Class ordering (must match the existing MobileNetV2 pipeline):
    The existing pipeline derives its 38 class names with
    ``sorted(os.listdir(split/train))``, i.e. Unicode code-point order. This
    module uses the same rule, so class index N means the same disease for both
    models.

Importing torch is done lazily inside functions so that the dataset-listing
helpers can also be imported from a TensorFlow-only environment (used by
``collect_mobilenetv2_predictions.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent

# Dataset switch. Default is the original PlantVillage experiment; setting the
# environment variable PDD_DATASET=plantwild points everything at the filtered
# 34-class PlantWild split (plantwild\split) and tags every cached feature /
# model / result file with "plantwild_" so the two experiments never collide.
DATASET = os.environ.get("PDD_DATASET", "plantvillage").lower()

if DATASET == "plantwild":
    SPLIT_DIR = PROJECT_ROOT / "plantwild" / "split"
    NUM_CLASSES = 34
    DATASET_TAG = "plantwild_"
elif DATASET == "plantvillage":
    SPLIT_DIR = PROJECT_ROOT / "split"
    NUM_CLASSES = 38
    DATASET_TAG = ""
else:
    raise ValueError(f"Unknown PDD_DATASET: {DATASET!r} (use 'plantvillage' or 'plantwild')")

TRAIN_DIR = SPLIT_DIR / "train"
VALIDATION_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"

# Real-world photographs (PlantDoc test split). Kept strictly separate from the
# PlantVillage train/validation/test splits so it can never be trained on.
REAL_WORLD_DIR = (
    PROJECT_ROOT
    / "real_world_dataset"
    / "real_world_dataset"
    / "PlantDoc-Dataset-master"
    / "test"
)

FEATURES_DIR = EXPERIMENT_DIR / "features"
MODELS_DIR = EXPERIMENT_DIR / "models"
RESULTS_DIR = EXPERIMENT_DIR / "results"

for _d in (FEATURES_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# ---------------------------------------------------------------------------
# DINOv2 backbone + preprocessing constants
# ---------------------------------------------------------------------------

# ViT-S/14: "small" ViT, patch size 14, embedding dimension 384.
BACKBONE_NAME = "dinov2_vits14"
BACKBONE_SOURCE = "facebookresearch/dinov2"
FEATURE_DIM = 384

# DINOv2 uses the standard ImageNet normalization, NOT the [0, 1] scaling used
# by the MobileNetV2 pipeline. Mixing the two silently degrades accuracy, so the
# constants are defined here once and reused by extraction, evaluation and
# single-image prediction.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2's own classification eval transform: resize the shorter side to 256,
# center-crop 224 (224 is a multiple of the patch size 14 -> 16x16 tokens).
DINOV2_RESIZE = 256
DINOV2_CROP = 224

# PlantDoc folder name -> PlantVillage class name (28 of the 38 classes have a
# PlantDoc equivalent). Mirrors config.PLANTDOC_MAPPING.
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


# ---------------------------------------------------------------------------
# Class discovery
# ---------------------------------------------------------------------------

def class_folders(directory: Path) -> List[str]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def class_names() -> List[str]:
    """The class names of the active dataset, in exact model output order."""
    names = class_folders(TRAIN_DIR)
    if len(names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(names)} in {TRAIN_DIR}")
    return names


def class_to_index() -> dict:
    return {name: i for i, name in enumerate(class_names())}


def image_files(directory: Path) -> List[str]:
    directory = Path(directory)
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Record listing (deterministic sorted order)
# ---------------------------------------------------------------------------

def plantvillage_records(directory: Path) -> List[Tuple[str, int]]:
    """Return [(absolute_path, class_index)] sorted by folder then filename."""
    mapping = class_to_index()
    records: List[Tuple[str, int]] = []
    directory = Path(directory)
    for folder in class_folders(directory):
        label = mapping[folder]
        for name in image_files(directory / folder):
            records.append((str(directory / folder / name), label))
    return records


def plantdoc_records(directory: Path) -> List[Tuple[str, int]]:
    """PlantDoc folder -> PlantVillage index, skipping unmapped folders."""
    mapping = class_to_index()
    records: List[Tuple[str, int]] = []
    directory = Path(directory)
    for folder in class_folders(directory):
        target = PLANTDOC_MAPPING.get(folder)
        if target is None:
            print(f"[warn] no PlantVillage mapping for PlantDoc folder: {folder}")
            continue
        label = mapping[target]
        for name in image_files(directory / folder):
            records.append((str(directory / folder / name), label))
    return records


def split_records(split: str) -> List[Tuple[str, int]]:
    if split == "train":
        return plantvillage_records(TRAIN_DIR)
    if split == "validation":
        return plantvillage_records(VALIDATION_DIR)
    if split == "test":
        return plantvillage_records(TEST_DIR)
    if split in ("realworld", "real_world", "plantdoc"):
        if DATASET != "plantvillage":
            raise ValueError(
                f"The PlantDoc real-world split is only defined for the "
                f"PlantVillage experiment (PDD_DATASET is {DATASET!r})."
            )
        return plantdoc_records(REAL_WORLD_DIR)
    raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------------
# DINOv2 backbone loading (frozen) + preprocessing
# ---------------------------------------------------------------------------

_BACKBONE_CACHE = {}


def load_backbone(device: str = "cpu"):
    """Load the frozen DINOv2 ViT-S/14 backbone.

    The first call downloads the backbone code from GitHub (via torch.hub) and
    the pretrained weights from dl.fbaipublicfiles.com. Both are cached under
    the torch hub directory, so later runs work offline. Only ``torch`` is
    required (the DINOv2 hubconf declares ``dependencies = ["torch"]``).
    """
    import torch

    key = (BACKBONE_NAME, device)
    if key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[key]

    model = torch.hub.load(BACKBONE_SOURCE, BACKBONE_NAME, trust_repo=True)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    _BACKBONE_CACHE[key] = model
    return model


def _resize_shorter_side(img, size: int):
    from PIL import Image

    width, height = img.size
    if width < height:
        new_w = size
        new_h = int(round(height * size / width))
    else:
        new_h = size
        new_w = int(round(width * size / height))
    return img.resize((new_w, new_h), Image.BICUBIC)


def preprocess_pil(img, resize: int = DINOV2_RESIZE, crop: int = DINOV2_CROP):
    """PIL image -> normalized float32 CHW tensor, ready for the DINOv2 backbone.

    Exactly matches DINOv2's own classification eval transform
    (Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)).
    """
    import torch

    img = img.convert("RGB")
    img = _resize_shorter_side(img, resize)

    width, height = img.size
    left = (width - crop) // 2
    top = (height - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))

    array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def extract_features_batch(backbone, batch_tensor, device: str = "cpu"):
    """Return a (B, FEATURE_DIM) numpy array of CLS-token features."""
    import torch

    batch_tensor = batch_tensor.to(device)
    with torch.no_grad():
        output = backbone(batch_tensor)

    if isinstance(output, dict):
        output = output.get("x_norm_clstoken", output)

    features = output.detach().cpu().float().numpy()
    if features.ndim == 3:
        features = features[:, 0, :]
    return features


def feature_cache_paths(split: str, backbone: str = BACKBONE_NAME,
                        crop: int = DINOV2_CROP):
    tag = f"{DATASET_TAG}{split}_{backbone}_{crop}"
    return {
        "features": FEATURES_DIR / f"{tag}.features.npy",
        "labels": FEATURES_DIR / f"{tag}.labels.npy",
        "paths": FEATURES_DIR / f"{tag}.paths.json",
        "meta": FEATURES_DIR / f"{tag}.meta.json",
    }


def artifact_path(filename: str) -> Path:
    """Model/result filename, prefixed with the dataset tag when needed."""
    return MODELS_DIR / f"{DATASET_TAG}{filename}"


def result_path(filename: str) -> Path:
    return RESULTS_DIR / f"{DATASET_TAG}{filename}"


def save_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
