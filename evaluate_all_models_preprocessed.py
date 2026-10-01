import os
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

PLANTDOC_DIR = (
    r"C:\Plant_disease_detection\real_world_dataset"
    r"\real_world_dataset\Plantdoc-Dataset-master"
    r"\plantdoc_preprocessed"
)

MODEL_PATHS = {
    "Basic CNN": os.path.join(
        BASE_DIR, "model", "plant_disease_model.keras"
    ),

    "Original ResNet18": os.path.join(
        BASE_DIR, "model", "resnet18_plant_disease.keras"
    ),

    "Improved ResNet18": os.path.join(
        BASE_DIR, "model", "resnet18_improved.keras"
    ),

    "MobileNetV2": os.path.join(
        BASE_DIR, "model", "mobilenetv2_plant_disease.keras"
    )
}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# PLANTVILLAGE CLASS NAMES
# IMPORTANT: ORDER MUST MATCH MODEL TRAINING
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ============================================================
# PLANTDOC FOLDER -> PLANTVILLAGE CLASS MAPPING
# ============================================================

PLANTDOC_MAPPING = {

    "Apple Scab Leaf":
        "Apple___Apple_scab",

    "Apple leaf":
        "Apple___healthy",

    "Apple rust leaf":
        "Apple___Cedar_apple_rust",

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

    "Tomato Septoria leaf spot":
        "Tomato___Septoria_leaf_spot",

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

    "grape leaf":
        "Grape___healthy",

    "grape leaf black rot":
        "Grape___Black_rot"
}


# ============================================================
# LOAD PLANTDOC IMAGES
# ============================================================

def load_plantdoc():

    images = []
    labels = []
    class_names = []
    filenames = []

    print("\nLoading preprocessed PlantDoc images...")

    for folder_name in sorted(os.listdir(PLANTDOC_DIR)):

        folder_path = os.path.join(PLANTDOC_DIR, folder_name)

        if not os.path.isdir(folder_path):
            continue

        if folder_name not in PLANTDOC_MAPPING:
            print("WARNING: No mapping found for:", folder_name)
            continue

        target_class = PLANTDOC_MAPPING[folder_name]

        if target_class not in CLASS_NAMES:
            print("WARNING: Target class not found:", target_class)
            continue

        label_index = CLASS_NAMES.index(target_class)

        for filename in sorted(os.listdir(folder_path)):

            file_path = os.path.join(folder_path, filename)

            try:

                image = Image.open(file_path).convert("RGB")
                image = image.resize(IMAGE_SIZE)

                image_array = np.array(image, dtype=np.float32)

                # Normalize exactly once
                image_array = image_array / 255.0

                images.append(image_array)
                labels.append(label_index)
                class_names.append(folder_name)
                filenames.append(filename)

            except Exception as e:

                print(
                    "Could not load:",
                    file_path,
                    "| Error:",
                    e
                )

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print("Images loaded:", len(images))

    return images, labels, class_names, filenames


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, model_path, images, labels, folder_names):

    print("\n" + "=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Running predictions...")

    predictions = model.predict(
        images,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    predicted_labels = np.argmax(predictions, axis=1)

    correct = np.sum(predicted_labels == labels)
    total = len(labels)

    accuracy = (correct / total) * 100

    print("\nOverall Result")
    print("-" * 70)
    print(f"Correct predictions : {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"Total images        : {total}")
    print(f"Accuracy            : {accuracy:.2f}%")

    # --------------------------------------------------------
    # PER-CLASS RESULTS
    # --------------------------------------------------------

    class_results = defaultdict(lambda: [0, 0])

    for actual, predicted, folder in zip(
        labels,
        predicted_labels,
        folder_names
    ):

        class_results[folder][1] += 1

        if actual == predicted:
            class_results[folder][0] += 1

    print("\nPer-Class Accuracy")
    print("-" * 70)

    for folder in sorted(class_results.keys()):

        correct_class = class_results[folder][0]
        total_class = class_results[folder][1]

        class_accuracy = (
            correct_class / total_class
        ) * 100

        print(
            f"{folder:<35} "
            f"{correct_class:>2}/{total_class:<2} "
            f"({class_accuracy:>6.2f}%)"
        )

    return accuracy, correct, total


# ============================================================
# MAIN
# ============================================================

print("\n")
print("=" * 70)
print("PLANTDOC PREPROCESSED EVALUATION")
print("ALL FOUR MODELS")
print("=" * 70)

# Check PlantDoc directory
if not os.path.exists(PLANTDOC_DIR):

    print("\nERROR: PlantDoc directory not found:")
    print(PLANTDOC_DIR)
    input("\nPress Enter to exit...")
    raise SystemExit


# Check model files
print("\nChecking model files...")

for model_name, model_path in MODEL_PATHS.items():

    if os.path.exists(model_path):

        print(f"[OK] {model_name}")

    else:

        print(f"[MISSING] {model_name}")
        print(model_path)

        input("\nPress Enter to exit...")
        raise SystemExit


# Load images only once
images, labels, folder_names, filenames = load_plantdoc()


# Check total
if len(images) == 0:

    print("\nERROR: No images were loaded.")
    input("\nPress Enter to exit...")
    raise SystemExit


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

results = {}

for model_name, model_path in MODEL_PATHS.items():

    accuracy, correct, total = evaluate_model(
        model_name,
        model_path,
        images,
        labels,
        folder_names
    )

    results[model_name] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<25}"
    f"{'Correct':>10}"
    f"{'Total':>10}"
    f"{'Accuracy':>15}"
)

print("-" * 70)

for model_name, result in results.items():

    print(
        f"{model_name:<25}"
        f"{result['correct']:>10}"
        f"{result['total']:>10}"
        f"{result['accuracy']:>14.2f}%"
    )

print("=" * 70)

print("\nEvaluation completed successfully.")
print("Dataset:", PLANTDOC_DIR)

input("\nPress Enter to exit...")