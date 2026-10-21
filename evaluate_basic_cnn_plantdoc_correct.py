import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\Plant_disease_detection\model\simple_cnn_plant_disease.keras"

PLANTDOC_DIR = (
    r"C:\Plant_disease_detection\real_world_dataset\real_world_dataset"
    r"\Plantdoc-Dataset-master\plantdoc_preprocessed"
)

RESULTS_PATH = (
    r"C:\Plant_disease_detection\model"
    r"\basic_cnn_plantdoc_corrected_results.json"
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# EXACT PLANTDOC -> PLANTVILLAGE MAPPING
# ============================================================

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

# ============================================================
# CHECK NUMBER OF MAPPED CLASSES
# ============================================================

print("=" * 75)
print("BASIC CNN - CORRECTED PLANTDOC EVALUATION")
print("=" * 75)

print("\nMapped PlantDoc classes:", len(PLANTDOC_TO_PLANTVILLAGE))

if len(PLANTDOC_TO_PLANTVILLAGE) != 27:
    raise ValueError(
        f"Expected 27 mappings, found {len(PLANTDOC_TO_PLANTVILLAGE)}"
    )

# ============================================================
# CHECK MODEL
# ============================================================

print("\nLoading Basic CNN...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully:")
print(MODEL_PATH)

# ============================================================
# CHECK PLANTDOC DIRECTORY
# ============================================================

if not os.path.exists(PLANTDOC_DIR):
    raise FileNotFoundError(
        f"PlantDoc directory not found:\n{PLANTDOC_DIR}"
    )

print("\nPlantDoc directory:")
print(PLANTDOC_DIR)

# ============================================================
# CHECK ALL 28 FOLDERS
# ============================================================

print("\n" + "=" * 75)
print("CHECKING THE 28 PLANTDOC CLASSES")
print("=" * 75)

missing_folders = []

for plantdoc_class in PLANTDOC_TO_PLANTVILLAGE:

    folder_path = os.path.join(
        PLANTDOC_DIR,
        plantdoc_class
    )

    if not os.path.isdir(folder_path):
        missing_folders.append(plantdoc_class)
        print(f"MISSING: {plantdoc_class}")
    else:
        print(f"OK: {plantdoc_class}")

if missing_folders:

    print("\nERROR: The following PlantDoc folders are missing:")

    for folder in missing_folders:
        print(" -", folder)

    raise SystemExit(
        "\nEvaluation stopped because one or more required "
        "PlantDoc folders are missing."
    )

# ============================================================
# LOAD PLANTVILLAGE CLASS ORDER
# ============================================================

# We use the exact class order from the PlantVillage test folder.
PLANTVILLAGE_TEST_DIR = r"C:\Plant_disease_detection\split\test"

pv_ds_for_classes = tf.keras.utils.image_dataset_from_directory(
    PLANTVILLAGE_TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

plantvillage_class_names = pv_ds_for_classes.class_names

print("\nPlantVillage classes:", len(plantvillage_class_names))

if len(plantvillage_class_names) != 38:
    raise ValueError(
        f"Expected 38 PlantVillage classes, "
        f"found {len(plantvillage_class_names)}"
    )

plantvillage_class_to_index = {
    name: index
    for index, name in enumerate(plantvillage_class_names)
}

# ============================================================
# VERIFY TARGET CLASSES
# ============================================================

print("\n" + "=" * 75)
print("VERIFYING CLASS MAPPINGS")
print("=" * 75)

for plantdoc_class, pv_class in PLANTDOC_TO_PLANTVILLAGE.items():

    if pv_class not in plantvillage_class_to_index:

        raise ValueError(
            f"\nPlantVillage class not found:\n"
            f"PlantDoc: {plantdoc_class}\n"
            f"Expected PlantVillage class: {pv_class}"
        )

    print(
        f"{plantdoc_class:35s} -> "
        f"{pv_class}"
    )

# ============================================================
# COLLECT EXACTLY 236 PLANTDOC IMAGES
# ============================================================

print("\n" + "=" * 75)
print("COLLECTING PLANTDOC IMAGES")
print("=" * 75)

image_paths = []
true_labels = []

class_counts = {}

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

for plantdoc_class, pv_class in PLANTDOC_TO_PLANTVILLAGE.items():

    folder_path = os.path.join(
        PLANTDOC_DIR,
        plantdoc_class
    )

    label = plantvillage_class_to_index[pv_class]

    count = 0

    for filename in sorted(os.listdir(folder_path)):

        filepath = os.path.join(
            folder_path,
            filename
        )

        if not os.path.isfile(filepath):
            continue

        if not filename.lower().endswith(valid_extensions):
            continue

        image_paths.append(filepath)
        true_labels.append(label)

        count += 1

    class_counts[plantdoc_class] = count

    print(
        f"{plantdoc_class:35s} : {count:3d}"
    )

# ============================================================
# CHECK TOTAL = 236
# ============================================================

total_images = len(image_paths)

print("\n" + "-" * 75)
print("TOTAL PLANTDOC IMAGES:", total_images)
print("-" * 75)

if total_images != 236:

    print("\nERROR!")
    print(
        f"The evaluation contains {total_images} images, "
        f"but exactly 236 images are required."
    )

    print("\nClass counts:")

    for class_name, count in class_counts.items():
        print(f"{class_name:35s} : {count}")

    raise SystemExit(
        "\nEvaluation STOPPED. "
        "No incorrect PlantDoc result will be reported."
    )

print("\nSUCCESS: Exactly 236 PlantDoc images found.")

# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("RUNNING BASIC CNN PREDICTIONS")
print("=" * 75)

y_true = []
y_pred = []

for start in range(
    0,
    len(image_paths),
    BATCH_SIZE
):

    batch_paths = image_paths[
        start:start + BATCH_SIZE
    ]

    batch_images = []

    for filepath in batch_paths:

        image = tf.keras.utils.load_img(
            filepath,
            target_size=IMG_SIZE,
            color_mode="rgb"
        )

        image_array = tf.keras.utils.img_to_array(
            image
        )

        # IMPORTANT:
        # Do NOT divide by 255 here.
        #
        # The genuine Basic CNN already contains:
        # Rescaling(1./255)
        #
        # Therefore raw 0-255 images are passed to
        # the model, exactly as during PlantVillage evaluation.

        batch_images.append(image_array)

    batch_images = np.array(
        batch_images,
        dtype=np.float32
    )

    predictions = model.predict(
        batch_images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        true_labels[
            start:start + BATCH_SIZE
        ]
    )

    y_pred.extend(
        predicted_labels
    )

    print(
        f"Processed {min(start + BATCH_SIZE, total_images)}"
        f"/{total_images}"
    )

# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ============================================================
# CALCULATE METRICS
# ============================================================

correct = int(
    np.sum(y_true == y_pred)
)

total = len(y_true)

accuracy = correct / total

precision = precision_score(
    y_true,
    y_pred,
    labels=list(
        set(y_true)
    ),
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    labels=list(
        set(y_true)
    ),
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    labels=list(
        set(y_true)
    ),
    average="macro",
    zero_division=0
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL BASIC CNN - PLANTDOC RESULTS")
print("=" * 75)

print(f"\nAccuracy        : {accuracy * 100:.2f}%")
print(f"Macro Precision : {precision * 100:.2f}%")
print(f"Macro Recall    : {recall * 100:.2f}%")
print(f"Macro F1-score  : {f1 * 100:.2f}%")
print(f"Correct         : {correct}/{total}")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "Basic CNN",
    "dataset": "PlantDoc",
    "number_of_classes": 28,
    "total_images": total,
    "correct": correct,
    "accuracy": round(accuracy * 100, 2),
    "macro_precision": round(precision * 100, 2),
    "macro_recall": round(recall * 100, 2),
    "macro_f1": round(f1 * 100, 2),
    "class_counts": class_counts
}

with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nResults saved to:")
print(RESULTS_PATH)

print("\n" + "=" * 75)
print("PLANTDOC EVALUATION COMPLETE")
print("=" * 75)