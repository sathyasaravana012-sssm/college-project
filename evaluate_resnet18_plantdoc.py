import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Plant_disease_detection\model\resnet18_plant_disease.keras"

# Change this if your PlantDoc folder is different
PLANTDOC_DIR = r"C:\Plant_disease_detection\real_world_dataset\real_world_dataset\Plantdoc-Dataset-master\test"

IMAGE_SIZE = (224, 224)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ResNet18 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# PLANTDOC → PLANTVILLAGE CLASS MAPPING
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

    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",

    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",

    "Squash Powdery mildew leaf":
        "Squash___Powdery_mildew",

    "Strawberry leaf": "Strawberry___healthy",

    "Tomato Early blight leaf":
        "Tomato___Early_blight",

    "Tomato Septoria leaf spot":
        "Tomato___Septoria_leaf_spot",

    "Tomato leaf": "Tomato___healthy",

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

    "grape leaf": "Grape___healthy",

    "grape leaf black rot":
        "Grape___Black_rot"
}

# ============================================================
# GET MODEL CLASS NAMES
# ============================================================

TRAIN_DIR = r"C:\Plant_disease_detection\split\train"

class_names = sorted([
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

class_to_index = {
    name: idx
    for idx, name in enumerate(class_names)
}

# ============================================================
# EVALUATION
# ============================================================

total = 0
correct = 0

print("\n" + "=" * 70)
print("PLANTDOC REAL-WORLD EVALUATION")
print("=" * 70)

for plantdoc_class, plantvillage_class in class_mapping.items():

    folder_path = os.path.join(
        PLANTDOC_DIR,
        plantdoc_class
    )

    if not os.path.exists(folder_path):

        print(f"{plantdoc_class:<35} Folder not found")
        continue

    true_label = class_to_index[plantvillage_class]

    class_total = 0
    class_correct = 0

    for file in os.listdir(folder_path):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

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

                # IMPORTANT
                image = image / 255.0

                image = np.expand_dims(
                    image,
                    axis=0
                )

                prediction = model.predict(
                    image,
                    verbose=0
                )

                predicted = np.argmax(
                    prediction
                )

                if predicted == true_label:

                    correct += 1
                    class_correct += 1

                total += 1
                class_total += 1

            except Exception as e:

                print("Error:", image_path)
                print(e)

    if class_total > 0:

        accuracy = (
            class_correct /
            class_total
        ) * 100

        print(
            f"{plantdoc_class:<35}"
            f"{class_correct}/{class_total} "
            f"({accuracy:.2f}%)"
        )

# ============================================================
# FINAL RESULT
# ============================================================

final_accuracy = (
    correct / total
) * 100

print("\n" + "=" * 70)
print("FINAL PLANTDOC RESULT")
print("=" * 70)

print("Images evaluated :", total)
print("Correct          :", correct)
print("Incorrect        :", total - correct)
print(
    f"Accuracy         : "
    f"{final_accuracy:.2f}%"
)

print("=" * 70)