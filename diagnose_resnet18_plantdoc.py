import os
from collections import Counter
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Plant_disease_detection\model\resnet18_plant_disease.keras"

PLANTDOC_DIR = r"C:\Plant_disease_detection\real_world_dataset\real_world_dataset\Plantdoc-Dataset-master\test"

TRAIN_DIR = r"C:\Plant_disease_detection\split\train"

IMAGE_SIZE = (224, 224)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ResNet18...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# GET EXACT PLANTVILLAGE CLASS NAMES
# ============================================================

class_names = sorted([
    folder
    for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

class_to_index = {
    name: i
    for i, name in enumerate(class_names)
}

print("\nNumber of PlantVillage classes:", len(class_names))

# ============================================================
# PLANTDOC → PLANTVILLAGE MAPPING
# ============================================================

class_mapping = {

    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",

    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",

    "Blueberry leaf": "Blueberry___healthy",

    "Cherry leaf": "Cherry_(including_sour)___healthy",

    "Corn Gray leaf spot":
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",

    "Corn leaf blight":
        "Corn_(maize)___Northern_Leaf_Blight",

    "Corn rust leaf":
        "Corn_(maize)___Common_rust_",

    "Peach leaf": "Peach___healthy",

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
# CHECK MAPPING
# ============================================================

print("\nChecking class mapping...")

for plantdoc_class, plantvillage_class in class_mapping.items():

    if plantvillage_class not in class_to_index:

        print(
            "WARNING:",
            plantdoc_class,
            "->",
            plantvillage_class,
            "NOT FOUND"
        )

print("\nMapping check complete.")

# ============================================================
# EVALUATION
# ============================================================

total = 0
correct = 0

print("\n" + "=" * 90)
print("RESNET18 PLANTDOC DIAGNOSTIC")
print("=" * 90)

for plantdoc_class, plantvillage_class in class_mapping.items():

    folder_path = os.path.join(
        PLANTDOC_DIR,
        plantdoc_class
    )

    if not os.path.exists(folder_path):

        print(
            f"{plantdoc_class:<35} FOLDER NOT FOUND"
        )

        continue

    if plantvillage_class not in class_to_index:

        print(
            f"{plantdoc_class:<35} INVALID MAPPING"
        )

        continue

    true_label = class_to_index[
        plantvillage_class
    ]

    class_total = 0
    class_correct = 0

    prediction_counter = Counter()

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for file in files:

        image_path = os.path.join(
            folder_path,
            file
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = image.resize(
                IMAGE_SIZE
            )

            image = np.array(
                image,
                dtype=np.float32
            )

            # SAME preprocessing used during ResNet18 training
            image = image / 255.0

            image = np.expand_dims(
                image,
                axis=0
            )

            prediction = model.predict(
                image,
                verbose=0
            )[0]

            predicted_label = int(
                np.argmax(prediction)
            )

            predicted_class = class_names[
                predicted_label
            ]

            prediction_counter[
                predicted_class
            ] += 1

            if predicted_label == true_label:

                class_correct += 1
                correct += 1

            class_total += 1
            total += 1

        except Exception as e:

            print(
                "\nERROR:",
                image_path
            )

            print(e)

    # --------------------------------------------------------
    # CLASS RESULT
    # --------------------------------------------------------

    if class_total > 0:

        accuracy = (
            class_correct /
            class_total
        ) * 100

        common_predictions = (
            prediction_counter
            .most_common(3)
        )

        prediction_text = ", ".join(
            [
                f"{name} ({count})"
                for name, count
                in common_predictions
            ]
        )

        print(
            f"\n{plantdoc_class}"
        )

        print(
            f"  Correct: "
            f"{class_correct}/{class_total}"
            f" ({accuracy:.2f}%)"
        )

        print(
            f"  Top predictions: "
            f"{prediction_text}"
        )

# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 90)
print("FINAL RESULT")
print("=" * 90)

print("Images evaluated :", total)
print("Correct          :", correct)
print("Incorrect        :", total - correct)

if total > 0:

    accuracy = (
        correct / total
    ) * 100

    print(
        f"Accuracy         : {accuracy:.2f}%"
    )

else:

    print("Accuracy         : No images evaluated")

print("=" * 90)