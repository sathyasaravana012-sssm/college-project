import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "mobilenetv2_plant_disease.keras"
)

PLANTDOC_DIR = os.path.join(
    BASE_DIR, "real_world_dataset", "real_world_dataset","PlantDoc-Dataset-master", "test"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# PLANTVILLAGE CLASS NAMES - SAME ORDER USED DURING TRAINING
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
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",

    "Strawberry leaf": "Strawberry___healthy",

    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus":
        "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus":
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",

    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot"
}

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("MOBILENETV2 - PLANTDOC REAL-WORLD EVALUATION")
print("=" * 70)

print("\nLoading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# LOAD PLANTDOC IMAGES
# ============================================================

images = []
true_labels = []
class_names_found = []

for folder_name in sorted(os.listdir(PLANTDOC_DIR)):

    folder_path = os.path.join(PLANTDOC_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    if folder_name not in PLANTDOC_MAPPING:
        print("WARNING: No mapping for:", folder_name)
        continue

    plantvillage_class = PLANTDOC_MAPPING[folder_name]

    if plantvillage_class not in CLASS_NAMES:
        print("WARNING: Class not found:", plantvillage_class)
        continue

    class_index = CLASS_NAMES.index(plantvillage_class)

    class_names_found.append(folder_name)

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)

        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)

            img_array = np.array(img, dtype=np.float32)

            # IMPORTANT:
            # MobileNetV2 was trained with pixel values scaled to 0-1
            img_array = img_array / 255.0

            images.append(img_array)
            true_labels.append(class_index)

        except Exception as e:
            print("Could not read:", file_path)
            print("Error:", e)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

images = np.array(images, dtype=np.float32)
true_labels = np.array(true_labels)

print("\nTotal PlantDoc images:", len(images))

# ============================================================
# PREDICTION
# ============================================================

print("\nRunning MobileNetV2 predictions...")

predictions = model.predict(
    images,
    batch_size=BATCH_SIZE,
    verbose=1
)

predicted_labels = np.argmax(predictions, axis=1)

# ============================================================
# OVERALL ACCURACY
# ============================================================

correct = np.sum(predicted_labels == true_labels)
total = len(true_labels)

accuracy = (correct / total) * 100

print("\n" + "=" * 70)
print("MOBILENETV2 - PLANTDOC RESULT")
print("=" * 70)

print(f"Correct Predictions : {correct}")
print(f"Incorrect Predictions : {total - correct}")
print(f"Total Images : {total}")
print(f"Accuracy : {accuracy:.2f}%")

print("=" * 70)

# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\nPER-CLASS RESULTS")
print("=" * 70)

start_index = 0

for folder_name in sorted(class_names_found):

    folder_path = os.path.join(PLANTDOC_DIR, folder_name)

    image_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            Image.open(file_path).convert("RGB")
            image_count += 1
        except:
            pass

    end_index = start_index + image_count

    class_true = true_labels[start_index:end_index]
    class_pred = predicted_labels[start_index:end_index]

    class_correct = np.sum(class_true == class_pred)

    class_accuracy = (
        class_correct / image_count * 100
        if image_count > 0 else 0
    )

    print(
        f"{folder_name:<40} "
        f"{class_correct}/{image_count} "
        f"({class_accuracy:.2f}%)"
    )

    start_index = end_index

print("=" * 70)
print("Evaluation completed.")
print("=" * 70)