import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"C:\Plant_disease_detection\model\simple_cnn_plant_disease.keras"

PLANTVILLAGE_TEST_DIR = r"C:\Plant_disease_detection\split\test"

# IMPORTANT:
# Change this ONLY if your PlantDoc preprocessed folder is somewhere else.
PLANTDOC_DIR = r"C:\Plant_disease_detection\real_world_dataset\real_world_dataset\Plantdoc-Dataset-master\plantdoc_preprocessed"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("BASIC CNN EVALUATION")
print("=" * 70)

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Model:", MODEL_PATH)

# ============================================================
# LOAD PLANTVILLAGE DATASET
# ============================================================

print("\n" + "=" * 70)
print("1. PLANTVILLAGE TEST SET - 38 CLASSES")
print("=" * 70)

plantvillage_ds = tf.keras.utils.image_dataset_from_directory(
    PLANTVILLAGE_TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

pv_class_names = plantvillage_ds.class_names

print("\nNumber of PlantVillage classes:", len(pv_class_names))
print("Classes:")
for i, name in enumerate(pv_class_names):
    print(f"{i:2d}: {name}")

# ============================================================
# PREDICT PLANTVILLAGE
# ============================================================

print("\nRunning PlantVillage predictions...")

y_true_pv = []
y_pred_pv = []

for images, labels in plantvillage_ds:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true_pv.extend(labels.numpy())
    y_pred_pv.extend(predicted_labels)

y_true_pv = np.array(y_true_pv)
y_pred_pv = np.array(y_pred_pv)

# ============================================================
# PLANTVILLAGE METRICS
# ============================================================

pv_correct = np.sum(y_true_pv == y_pred_pv)
pv_total = len(y_true_pv)
pv_accuracy = pv_correct / pv_total

pv_precision = precision_score(
    y_true_pv,
    y_pred_pv,
    average="macro",
    zero_division=0
)

pv_recall = recall_score(
    y_true_pv,
    y_pred_pv,
    average="macro",
    zero_division=0
)

pv_f1 = f1_score(
    y_true_pv,
    y_pred_pv,
    average="macro",
    zero_division=0
)

print("\n" + "-" * 70)
print("PLANTVILLAGE RESULTS")
print("-" * 70)

print(f"Accuracy        : {pv_accuracy * 100:.2f}%")
print(f"Correct         : {pv_correct}")
print(f"Total           : {pv_total}")
print(f"Macro Precision : {pv_precision * 100:.2f}%")
print(f"Macro Recall    : {pv_recall * 100:.2f}%")
print(f"Macro F1-score  : {pv_f1 * 100:.2f}%")

# ============================================================
# CHECK PLANTDOC DIRECTORY
# ============================================================

print("\n" + "=" * 70)
print("2. PLANTDOC REAL-WORLD EVALUATION")
print("=" * 70)

if not os.path.exists(PLANTDOC_DIR):
    print("\nERROR: PlantDoc directory was not found:")
    print(PLANTDOC_DIR)
    print("\nPlease change PLANTDOC_DIR at the top of this script.")
    raise SystemExit

# ============================================================
# GET PLANTDOC CLASS FOLDERS
# ============================================================

plantdoc_classes = sorted([
    folder
    for folder in os.listdir(PLANTDOC_DIR)
    if os.path.isdir(os.path.join(PLANTDOC_DIR, folder))
])

print("\nPlantDoc classes found:", len(plantdoc_classes))

for i, name in enumerate(plantdoc_classes):
    print(f"{i:2d}: {name}")

# ============================================================
# MAP PLANTDOC CLASS NAMES TO PLANTVILLAGE CLASS NAMES
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
# PlantDoc folder names may not exactly match PlantVillage
# class names.
#
# Add/edit mappings here if required.
# ------------------------------------------------------------

PLANTDOC_TO_PLANTVILLAGE = {

    # Apple
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",

    # Bell Pepper
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",

    # Cherry
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Cherry leaf spot": "Cherry_(including_sour)___Powdery_mildew",

    # Corn
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Corn healthy leaf": "Corn_(maize)___healthy",

    # Grape
    "Grape leaf": "Grape___healthy",
    "Grape leaf black rot": "Grape___Black_rot",
    "Grape leaf esca": "Grape___Esca_(Black_Measles)",
    "Grape leaf isariopsis": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",

    # Peach
    "Peach leaf": "Peach___healthy",
    "Peach leaf spot": "Peach___Bacterial_spot",

    # Potato
    "Potato leaf": "Potato___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",

    # Tomato
    "Tomato leaf": "Tomato___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Late blight leaf": "Tomato___Late_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf septoria": "Tomato___Septoria_leaf_spot",
    "Tomato leaf spider mites": "Tomato___Spider_mites Two-spotted_spider_mite",

}

# ============================================================
# CREATE PLANTDOC DATASET MANUALLY
# ============================================================

image_paths = []
true_labels = []

plantvillage_class_to_index = {
    name: index
    for index, name in enumerate(pv_class_names)
}

print("\nChecking PlantDoc class mapping...")

for plantdoc_class in plantdoc_classes:

    if plantdoc_class not in PLANTDOC_TO_PLANTVILLAGE:
        print(f"WARNING: No mapping for PlantDoc class: {plantdoc_class}")
        continue

    pv_class = PLANTDOC_TO_PLANTVILLAGE[plantdoc_class]

    if pv_class not in plantvillage_class_to_index:
        print(
            f"WARNING: PlantVillage class not found for "
            f"{plantdoc_class} -> {pv_class}"
        )
        continue

    label = plantvillage_class_to_index[pv_class]

    class_dir = os.path.join(PLANTDOC_DIR, plantdoc_class)

    for filename in os.listdir(class_dir):

        filepath = os.path.join(class_dir, filename)

        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            image_paths.append(filepath)
            true_labels.append(label)

print("\nPlantDoc images selected:", len(image_paths))

if len(image_paths) == 0:
    print("\nERROR: No PlantDoc images were found.")
    raise SystemExit

# ============================================================
# PREDICT PLANTDOC
# ============================================================

print("\nRunning PlantDoc predictions...")

y_true_pd = []
y_pred_pd = []

for start in range(0, len(image_paths), BATCH_SIZE):

    batch_paths = image_paths[start:start + BATCH_SIZE]

    batch_images = []

    for filepath in batch_paths:

        image = tf.keras.utils.load_img(
            filepath,
            target_size=IMG_SIZE
        )

        image_array = tf.keras.utils.img_to_array(image)

        batch_images.append(image_array)

    batch_images = np.array(batch_images, dtype=np.float32)

    predictions = model.predict(
        batch_images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true_pd.extend(
        true_labels[start:start + BATCH_SIZE]
    )

    y_pred_pd.extend(
        predicted_labels
    )

y_true_pd = np.array(y_true_pd)
y_pred_pd = np.array(y_pred_pd)

# ============================================================
# PLANTDOC METRICS
# ============================================================

pd_correct = np.sum(y_true_pd == y_pred_pd)
pd_total = len(y_true_pd)

pd_accuracy = pd_correct / pd_total

pd_precision = precision_score(
    y_true_pd,
    y_pred_pd,
    average="macro",
    zero_division=0
)

pd_recall = recall_score(
    y_true_pd,
    y_pred_pd,
    average="macro",
    zero_division=0
)

pd_f1 = f1_score(
    y_true_pd,
    y_pred_pd,
    average="macro",
    zero_division=0
)

print("\n" + "-" * 70)
print("PLANTDOC RESULTS")
print("-" * 70)

print(f"Accuracy        : {pd_accuracy * 100:.2f}%")
print(f"Correct         : {pd_correct}")
print(f"Total           : {pd_total}")
print(f"Macro Precision : {pd_precision * 100:.2f}%")
print(f"Macro Recall    : {pd_recall * 100:.2f}%")
print(f"Macro F1-score  : {pd_f1 * 100:.2f}%")

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "Basic CNN",
    "plantvillage": {
        "accuracy": round(pv_accuracy * 100, 2),
        "correct": int(pv_correct),
        "total": int(pv_total),
        "macro_precision": round(pv_precision * 100, 2),
        "macro_recall": round(pv_recall * 100, 2),
        "macro_f1": round(pv_f1 * 100, 2)
    },
    "plantdoc": {
        "accuracy": round(pd_accuracy * 100, 2),
        "correct": int(pd_correct),
        "total": int(pd_total),
        "macro_precision": round(pd_precision * 100, 2),
        "macro_recall": round(pd_recall * 100, 2),
        "macro_f1": round(pd_f1 * 100, 2)
    }
}

RESULTS_PATH = r"C:\Plant_disease_detection\model\basic_cnn_final_evaluation.json"

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL BASIC CNN SUMMARY")
print("=" * 70)

print("\nPlantVillage:")
print(f"  Accuracy        : {pv_accuracy * 100:.2f}%")
print(f"  Macro Precision : {pv_precision * 100:.2f}%")
print(f"  Macro Recall    : {pv_recall * 100:.2f}%")
print(f"  Macro F1        : {pv_f1 * 100:.2f}%")
print(f"  Correct         : {pv_correct}/{pv_total}")

print("\nPlantDoc:")
print(f"  Accuracy        : {pd_accuracy * 100:.2f}%")
print(f"  Macro Precision : {pd_precision * 100:.2f}%")
print(f"  Macro Recall    : {pd_recall * 100:.2f}%")
print(f"  Macro F1        : {pd_f1 * 100:.2f}%")
print(f"  Correct         : {pd_correct}/{pd_total}")

print("\nResults saved to:")
print(RESULTS_PATH)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)