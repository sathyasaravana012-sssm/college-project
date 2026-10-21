# ================================================================
# FINAL THREE-MODEL ENSEMBLE EVALUATION
# Basic CNN + ResNet18 Improved + MobileNetV2
#
# PlantVillage : 38 classes / 8176 test images
# PlantDoc     : 27 common classes / 236 images
#
# Ensemble     : Soft voting / probability averaging
#
# IMPORTANT:
#   Basic CNN contains its own Rescaling(1./255) layer.
#   Therefore Basic CNN receives 0-255 image values.
#
#   ResNet18 and MobileNetV2 receive images scaled to 0-1,
#   matching the preprocessing used by the evaluation pipeline.
# ================================================================

import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ================================================================
# PATHS
# ================================================================

PROJECT_DIR = r"C:\Plant_disease_detection"

MODEL_DIR = os.path.join(PROJECT_DIR, "model")

BASIC_CNN_PATH = os.path.join(
    MODEL_DIR,
    "simple_cnn_plant_disease.keras"
)

RESNET18_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_improved.keras"
)

MOBILENETV2_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv2_plant_disease.keras"
)

PLANTVILLAGE_TEST_DIR = os.path.join(
    PROJECT_DIR,
    "split",
    "test"
)

PLANTDOC_DIR = os.path.join(
    PROJECT_DIR,
    "real_world_dataset",
    "real_world_dataset",
    "Plantdoc-Dataset-master",
    "plantdoc_preprocessed"
)

OUTPUT_JSON = os.path.join(
    MODEL_DIR,
    "final_ensemble_verified_results.json"
)

# ================================================================
# SETTINGS
# ================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38

# ================================================================
# PLANTDOC -> PLANTVILLAGE MAPPING
#
# These are the 27 folders actually present in the user's
# PlantDoc evaluation dataset.
# ================================================================

PLANTDOC_TO_PLANTVILLAGE = {

    "Apple leaf":
        "Apple___healthy",

    "Apple rust leaf":
        "Apple___Cedar_apple_rust",

    "Apple Scab Leaf":
        "Apple___Apple_scab",

    "Bell_pepper leaf":
        "Pepper,_bell___healthy",

    "Bell_pepper leaf spot":
        "Pepper,_bell___Bacterial_spot",

    "Blueberry leaf":
        "Blueberry___healthy",

    "Cherry leaf":
        "Cherry_(including_sour)___healthy",

    "Corn Gray leaf spot":
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn leaf blight":
        "Corn_(maize)___Northern_Leaf_Blight",

    "Corn rust leaf":
        "Corn_(maize)___Common_rust_",

    "grape leaf":
        "Grape___healthy",

    "grape leaf black rot":
        "Grape___Black_rot",

    "Peach leaf":
        "Peach___healthy",

    "Potato leaf early blight":
        "Potato___Early_blight",

    "Potato leaf late blight":
        "Potato___Late_blight",

    "Raspberry leaf":
        "Raspberry___healthy",

    "Soyabean leaf":
        "Soybean___healthy",

    "Squash Powdery mildew leaf":
        "Squash___Powdery_mildew",

    "Strawberry leaf":
        "Strawberry___healthy",

    "Tomato Early blight leaf":
        "Tomato___Early_blight",

    "Tomato leaf":
        "Tomato___healthy",

    "Tomato leaf bacterial spot":
        "Tomato___Bacterial_spot",

    "Tomato leaf late blight":
        "Tomato___Late_blight",

    "Tomato leaf mosaic virus":
        "Tomato___Tomato_mosaic_virus",

    "Tomato leaf yellow virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",

    "Tomato mold leaf":
        "Tomato___Leaf_Mold",

    "Tomato Septoria leaf spot":
        "Tomato___Septoria_leaf_spot",
}

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def get_image_files(directory):
    """
    Recursively collect supported image files.
    """
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    files = []

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()

            if ext in extensions:
                files.append(
                    os.path.join(root, filename)
                )

    return sorted(files)


def load_image_raw(path):
    """
    Load image and return float32 array in 0-255 range.
    This is used for the Basic CNN because the Basic CNN
    contains its own Rescaling(1./255) layer.
    """

    image = tf.keras.utils.load_img(
        path,
        target_size=IMG_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    return image.astype(np.float32)


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy and macro-averaged metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "correct": int(np.sum(np.array(y_true) == np.array(y_pred))),
        "total": int(len(y_true))
    }


def print_metrics(title, metrics):
    """
    Print metrics in percentage form.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Accuracy        : {metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision : {metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Macro Recall    : {metrics['recall'] * 100:.2f}%"
    )

    print(
        f"Macro F1-score  : {metrics['f1'] * 100:.2f}%"
    )

    print(
        f"Correct         : "
        f"{metrics['correct']}/{metrics['total']}"
    )


# ================================================================
# START
# ================================================================

print("=" * 80)
print("FINAL THREE-MODEL ENSEMBLE EVALUATION")
print("=" * 80)

print()
print("Models:")
print("1. Basic CNN")
print("2. ResNet18 Improved")
print("3. MobileNetV2")

print()
print("Ensemble method:")
print("Soft voting / probability averaging")

# ================================================================
# CHECK PATHS
# ================================================================

print("\n" + "=" * 80)
print("CHECKING FILES")
print("=" * 80)

required_paths = {
    "Basic CNN model": BASIC_CNN_PATH,
    "ResNet18 model": RESNET18_PATH,
    "MobileNetV2 model": MOBILENETV2_PATH,
    "PlantVillage test": PLANTVILLAGE_TEST_DIR,
    "PlantDoc directory": PLANTDOC_DIR
}

for name, path in required_paths.items():

    if os.path.exists(path):

        print(f"OK: {name}")
        print(f"    {path}")

    else:

        raise FileNotFoundError(
            f"\nERROR: {name} not found:\n{path}"
        )

# ================================================================
# LOAD MODELS
# ================================================================

print("\n" + "=" * 80)
print("LOADING MODELS")
print("=" * 80)

print("\nLoading Basic CNN...")

basic_cnn = load_model(
    BASIC_CNN_PATH,
    compile=False
)

print("Basic CNN loaded.")

print("\nLoading ResNet18 Improved...")

resnet18 = load_model(
    RESNET18_PATH,
    compile=False
)

print("ResNet18 Improved loaded.")

print("\nLoading MobileNetV2...")

mobilenetv2 = load_model(
    MOBILENETV2_PATH,
    compile=False
)

print("MobileNetV2 loaded.")

# ================================================================
# VERIFY OUTPUT DIMENSIONS
# ================================================================

print("\n" + "=" * 80)
print("CHECKING MODEL OUTPUT DIMENSIONS")
print("=" * 80)

print(
    f"Basic CNN          : {basic_cnn.output_shape}"
)

print(
    f"ResNet18 Improved  : {resnet18.output_shape}"
)

print(
    f"MobileNetV2        : {mobilenetv2.output_shape}"
)

if basic_cnn.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        "Basic CNN does not output 38 classes."
    )

if resnet18.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        "ResNet18 does not output 38 classes."
    )

if mobilenetv2.output_shape[-1] != NUM_CLASSES:
    raise ValueError(
        "MobileNetV2 does not output 38 classes."
    )

print("\nAll three models output 38 classes.")

# ================================================================
# PLANTVILLAGE CLASS ORDER
# ================================================================

print("\n" + "=" * 80)
print("LOADING PLANTVILLAGE CLASS ORDER")
print("=" * 80)

plantvillage_dataset = tf.keras.utils.image_dataset_from_directory(
    PLANTVILLAGE_TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

plantvillage_class_names = (
    plantvillage_dataset.class_names
)

print(
    f"\nPlantVillage classes: "
    f"{len(plantvillage_class_names)}"
)

if len(plantvillage_class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} PlantVillage classes, "
        f"found {len(plantvillage_class_names)}"
    )

# ================================================================
# VERIFY PLANTDOC CLASSES
# ================================================================

print("\n" + "=" * 80)
print("CHECKING PLANTDOC CLASSES")
print("=" * 80)

plantdoc_classes = []

for folder in sorted(
    os.listdir(PLANTDOC_DIR)
):

    full_path = os.path.join(
        PLANTDOC_DIR,
        folder
    )

    if os.path.isdir(full_path):

        plantdoc_classes.append(folder)

print(
    f"\nPlantDoc mapped classes found: "
    f"{len(plantdoc_classes)}"
)

for folder in plantdoc_classes:

    if folder not in PLANTDOC_TO_PLANTVILLAGE:

        print(
            f"WARNING: No mapping found for: {folder}"
        )

print("\nPlantDoc mapping:")

for folder in plantdoc_classes:

    if folder in PLANTDOC_TO_PLANTVILLAGE:

        print(
            f"{folder:<35} -> "
            f"{PLANTDOC_TO_PLANTVILLAGE[folder]}"
        )

# ================================================================
# CREATE PLANTDOC CLASS INDEX
# ================================================================

plantdoc_valid_classes = []

for folder in plantdoc_classes:

    if folder in PLANTDOC_TO_PLANTVILLAGE:

        plantdoc_valid_classes.append(folder)

if len(plantdoc_valid_classes) != 27:

    raise ValueError(
        f"Expected 27 mapped PlantDoc classes, "
        f"found {len(plantdoc_valid_classes)}"
    )

plantdoc_class_to_index = {
    class_name: index
    for index, class_name
    in enumerate(plantdoc_valid_classes)
}

# ================================================================
# PLANTDOC IMAGE COLLECTION
# ================================================================

print("\n" + "=" * 80)
print("COLLECTING PLANTDOC IMAGES")
print("=" * 80)

plantdoc_paths = []
plantdoc_labels = []

for class_name in plantdoc_valid_classes:

    class_dir = os.path.join(
        PLANTDOC_DIR,
        class_name
    )

    files = get_image_files(
        class_dir
    )

    print(
        f"{class_name:<35}: "
        f"{len(files):>3}"
    )

    for file_path in files:

        plantdoc_paths.append(
            file_path
        )

        plantdoc_labels.append(
            plantdoc_class_to_index[class_name]
        )

print("\n" + "-" * 70)

print(
    f"TOTAL PLANTDOC IMAGES: "
    f"{len(plantdoc_paths)}"
)

print("-" * 70)

if len(plantdoc_paths) != 236:

    raise ValueError(
        f"Expected exactly 236 PlantDoc images, "
        f"found {len(plantdoc_paths)}"
    )

print(
    "\nSUCCESS: Exactly 236 PlantDoc images found."
)

# ================================================================
# HELPER FOR PLANTVILLAGE BATCH PREDICTION
# ================================================================

def predict_three_models(images_0_255):

    """
    Predict using all three models with model-specific
    preprocessing.

    Basic CNN:
        receives 0-255 because it contains Rescaling(1./255).

    ResNet18:
        receives 0-1.

    MobileNetV2:
        receives 0-1.
    """

    # ------------------------------------------------------------
    # Basic CNN
    # ------------------------------------------------------------

    basic_input = images_0_255

    basic_probabilities = basic_cnn.predict(
        basic_input,
        verbose=0
    )

    # ------------------------------------------------------------
    # ResNet18
    # ------------------------------------------------------------

    resnet_input = (
        images_0_255 / 255.0
    )

    resnet_probabilities = resnet18.predict(
        resnet_input,
        verbose=0
    )

    # ------------------------------------------------------------
    # MobileNetV2
    # ------------------------------------------------------------

    mobile_input = (
        images_0_255 / 255.0
    )

    mobile_probabilities = mobilenetv2.predict(
        mobile_input,
        verbose=0
    )

    return (
        basic_probabilities,
        resnet_probabilities,
        mobile_probabilities
    )


# ================================================================
# PLANTVILLAGE EVALUATION
# ================================================================

print("\n" + "=" * 80)
print("PLANTVILLAGE INDIVIDUAL + ENSEMBLE EVALUATION")
print("=" * 80)

pv_true = []
pv_basic_pred = []
pv_resnet_pred = []
pv_mobile_pred = []
pv_ensemble_pred = []

pv_total = 0

for batch_images, batch_labels in plantvillage_dataset:

    images = batch_images.numpy().astype(
        np.float32
    )

    labels = batch_labels.numpy()

    (
        basic_prob,
        resnet_prob,
        mobile_prob
    ) = predict_three_models(images)

    # Individual predictions

    basic_pred = np.argmax(
        basic_prob,
        axis=1
    )

    resnet_pred = np.argmax(
        resnet_prob,
        axis=1
    )

    mobile_pred = np.argmax(
        mobile_prob,
        axis=1
    )

    # ------------------------------------------------------------
    # SOFT VOTING
    # ------------------------------------------------------------

    ensemble_prob = (
        basic_prob
        + resnet_prob
        + mobile_prob
    ) / 3.0

    ensemble_pred = np.argmax(
        ensemble_prob,
        axis=1
    )

    pv_true.extend(
        labels.tolist()
    )

    pv_basic_pred.extend(
        basic_pred.tolist()
    )

    pv_resnet_pred.extend(
        resnet_pred.tolist()
    )

    pv_mobile_pred.extend(
        mobile_pred.tolist()
    )

    pv_ensemble_pred.extend(
        ensemble_pred.tolist()
    )

    pv_total += len(labels)

    if pv_total % 320 == 0:

        print(
            f"Processed "
            f"{pv_total}/8176"
        )

# ================================================================
# PLANTVILLAGE METRICS
# ================================================================

pv_basic_metrics = calculate_metrics(
    pv_true,
    pv_basic_pred
)

pv_resnet_metrics = calculate_metrics(
    pv_true,
    pv_resnet_pred
)

pv_mobile_metrics = calculate_metrics(
    pv_true,
    pv_mobile_pred
)

pv_ensemble_metrics = calculate_metrics(
    pv_true,
    pv_ensemble_pred
)

print_metrics(
    "PLANTVILLAGE - BASIC CNN",
    pv_basic_metrics
)

print_metrics(
    "PLANTVILLAGE - RESNET18 IMPROVED",
    pv_resnet_metrics
)

print_metrics(
    "PLANTVILLAGE - MOBILENETV2",
    pv_mobile_metrics
)

print_metrics(
    "PLANTVILLAGE - THREE-MODEL ENSEMBLE",
    pv_ensemble_metrics
)

# ================================================================
# PLANTDOC EVALUATION
# ================================================================

print("\n" + "=" * 80)
print("PLANTDOC INDIVIDUAL + ENSEMBLE EVALUATION")
print("=" * 80)

print(
    "\nEvaluating 236 images across "
    "27 common classes..."
)

pd_true = []
pd_basic_pred = []
pd_resnet_pred = []
pd_mobile_pred = []
pd_ensemble_pred = []

for index, image_path in enumerate(
    plantdoc_paths,
    start=1
):

    # ------------------------------------------------------------
    # LOAD IMAGE
    # ------------------------------------------------------------

    image = load_image_raw(
        image_path
    )

    # Add batch dimension

    image_batch = np.expand_dims(
        image,
        axis=0
    )

    # ------------------------------------------------------------
    # BASIC CNN
    # ------------------------------------------------------------

    basic_prob = basic_cnn.predict(
        image_batch,
        verbose=0
    )[0]

    # ------------------------------------------------------------
    # RESNET18
    # ------------------------------------------------------------

    resnet_prob = resnet18.predict(
        image_batch / 255.0,
        verbose=0
    )[0]

    # ------------------------------------------------------------
    # MOBILENETV2
    # ------------------------------------------------------------

    mobile_prob = mobilenetv2.predict(
        image_batch / 255.0,
        verbose=0
    )[0]

    # ------------------------------------------------------------
    # INDIVIDUAL PREDICTIONS
    # ------------------------------------------------------------

    basic_prediction = int(
        np.argmax(basic_prob)
    )

    resnet_prediction = int(
        np.argmax(resnet_prob)
    )

    mobile_prediction = int(
        np.argmax(mobile_prob)
    )

    # ------------------------------------------------------------
    # SOFT VOTING
    # ------------------------------------------------------------

    ensemble_prob = (
        basic_prob
        + resnet_prob
        + mobile_prob
    ) / 3.0

    ensemble_prediction = int(
        np.argmax(ensemble_prob)
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # PlantDoc labels are based on the 27 mapped classes.
    # The model outputs 38 PlantVillage classes.
    #
    # Therefore, a prediction is correct only when the predicted
    # PlantVillage class exactly matches the PlantVillage class
    # corresponding to the PlantDoc folder.
    # ------------------------------------------------------------

    plantdoc_folder = os.path.basename(
        os.path.dirname(image_path)
    )

    expected_plantvillage_class = (
        PLANTDOC_TO_PLANTVILLAGE[
            plantdoc_folder
        ]
    )

    expected_pv_index = (
        plantvillage_class_names.index(
            expected_plantvillage_class
        )
    )

    pd_true.append(
        expected_pv_index
    )

    pd_basic_pred.append(
        basic_prediction
    )

    pd_resnet_pred.append(
        resnet_prediction
    )

    pd_mobile_pred.append(
        mobile_prediction
    )

    pd_ensemble_pred.append(
        ensemble_prediction
    )

    if (
        index % 32 == 0
        or index == len(plantdoc_paths)
    ):

        print(
            f"Processed "
            f"{index}/{len(plantdoc_paths)}"
        )

# ================================================================
# PLANTDOC METRICS
#
# Macro metrics are calculated over the 38 PlantVillage output
# classes, using the mapped PlantDoc ground-truth labels.
#
# This is consistent with evaluating the original 38-class
# classifier on real-world images.
# ================================================================

pd_basic_metrics = calculate_metrics(
    pd_true,
    pd_basic_pred
)

pd_resnet_metrics = calculate_metrics(
    pd_true,
    pd_resnet_pred
)

pd_mobile_metrics = calculate_metrics(
    pd_true,
    pd_mobile_pred
)

pd_ensemble_metrics = calculate_metrics(
    pd_true,
    pd_ensemble_pred
)

print_metrics(
    "PLANTDOC - BASIC CNN",
    pd_basic_metrics
)

print_metrics(
    "PLANTDOC - RESNET18 IMPROVED",
    pd_resnet_metrics
)

print_metrics(
    "PLANTDOC - MOBILENETV2",
    pd_mobile_metrics
)

print_metrics(
    "PLANTDOC - THREE-MODEL ENSEMBLE",
    pd_ensemble_metrics
)

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nPLANTVILLAGE")
print("-" * 80)

print(
    f"Basic CNN          : "
    f"{pv_basic_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"ResNet18 Improved  : "
    f"{pv_resnet_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"MobileNetV2        : "
    f"{pv_mobile_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"ENSEMBLE           : "
    f"{pv_ensemble_metrics['accuracy'] * 100:.2f}%"
)

print("\nPLANTDOC")
print("-" * 80)

print(
    f"Basic CNN          : "
    f"{pd_basic_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"ResNet18 Improved  : "
    f"{pd_resnet_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"MobileNetV2        : "
    f"{pd_mobile_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"ENSEMBLE           : "
    f"{pd_ensemble_metrics['accuracy'] * 100:.2f}%"
)

# ================================================================
# SAVE RESULTS
# ================================================================

results = {

    "ensemble_method":
        "soft voting / probability averaging",

    "models": [
        "Basic CNN",
        "ResNet18 Improved",
        "MobileNetV2"
    ],

    "plantvillage": {

        "dataset":
            "PlantVillage",

        "classes":
            38,

        "total_images":
            len(pv_true),

        "basic_cnn":
            pv_basic_metrics,

        "resnet18_improved":
            pv_resnet_metrics,

        "mobilenetv2":
            pv_mobile_metrics,

        "ensemble":
            pv_ensemble_metrics
    },

    "plantdoc": {

        "dataset":
            "PlantDoc",

        "mapped_classes":
            27,

        "total_images":
            len(pd_true),

        "basic_cnn":
            pd_basic_metrics,

        "resnet18_improved":
            pd_resnet_metrics,

        "mobilenetv2":
            pd_mobile_metrics,

        "ensemble":
            pd_ensemble_metrics
    }
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4
    )

print("\n" + "=" * 80)

print(
    "Results saved to:"
)

print(
    OUTPUT_JSON
)

print("=" * 80)

print(
    "\nFINAL ENSEMBLE EVALUATION COMPLETE."
)