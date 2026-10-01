import os
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import Counter

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet18_improved.keras"
)

PLANTDOC_DIR = os.path.join(
    BASE_DIR,
    "real_world_dataset",
    "real_world_dataset",
    "Plantdoc-Dataset-master",
    "test"
)

IMAGE_SIZE = (224, 224)

# ============================================================
# PLANTVILLAGE CLASS NAMES
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
# PLANTDOC → PLANTVILLAGE MAPPING
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
# LOAD MODEL
# ============================================================

print("\nLoading Improved ResNet18...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# EVALUATE
# ============================================================

total = 0
correct = 0

all_true = []
all_pred = []

print("\n" + "=" * 75)
print("IMPROVED RESNET18 - PLANTDOC EVALUATION")
print("=" * 75)

for folder_name in sorted(os.listdir(PLANTDOC_DIR)):

    folder_path = os.path.join(
        PLANTDOC_DIR,
        folder_name
    )

    if not os.path.isdir(folder_path):
        continue

    if folder_name not in PLANTDOC_MAPPING:
        print(f"\nSkipping: {folder_name}")
        continue

    expected_class = PLANTDOC_MAPPING[folder_name]
    expected_index = CLASS_NAMES.index(expected_class)

    class_total = 0
    class_correct = 0

    predictions = []

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(
            folder_path,
            filename
        )

        try:

            image = Image.open(image_path).convert("RGB")

            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(
                image,
                dtype=np.float32
            )

            # Same preprocessing used by the model
            image_array = image_array / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            predictions.append(predicted_class)

            all_true.append(expected_index)
            all_pred.append(predicted_index)

            class_total += 1
            total += 1

            if predicted_index == expected_index:
                class_correct += 1
                correct += 1

        except Exception as e:

            print(
                f"Error: {image_path}"
            )
            print(e)

    class_accuracy = (
        class_correct / class_total * 100
        if class_total > 0
        else 0
    )

    print(
        f"{folder_name:<35}"
        f"{class_correct:>3}/{class_total:<3}"
        f" ({class_accuracy:>6.2f}%)"
    )

    if predictions:

        common = Counter(
            predictions
        ).most_common(3)

        print(
            "   Top predictions:",
            common
        )

# ============================================================
# FINAL RESULT
# ============================================================

final_accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

print("\n" + "=" * 75)
print("FINAL RESULT")
print("=" * 75)

print(f"Images evaluated : {total}")
print(f"Correct          : {correct}")
print(f"Incorrect        : {total - correct}")
print(f"Accuracy         : {final_accuracy:.2f}%")

print("=" * 75)