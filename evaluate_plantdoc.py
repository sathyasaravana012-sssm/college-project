import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Plant_disease_detection\model\plant_disease_model.keras"

PLANTDOC_PATH = r"C:\Plant_disease_detection\real_world_dataset\real_world_dataset\Plantdoc-Dataset-master\test"

IMAGE_SIZE = (224, 224)

# ============================================================
# PLANTVILLAGE CLASS ORDER
# This must match the order used when your model was trained.
# ============================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca",
    "Grape___Leaf_blight",
    "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper___Bacterial_spot",
    "Pepper___healthy",
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
    "Tomato___Spider_mites",
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

    "Bell_pepper leaf": "Pepper___healthy",
    "Bell_pepper leaf spot": "Pepper___Bacterial_spot",

    "Blueberry leaf": "Blueberry___healthy",

    "Cherry leaf": "Cherry___healthy",

    "Corn Gray leaf spot": "Corn___Cercospora_leaf_spot",
    "Corn leaf blight": "Corn___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn___Common_rust",

    "Peach leaf": "Peach___healthy",

    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",

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
    "Tomato mold leaf": "Tomato___Leaf_Mold"
}

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Number of output classes:", model.output_shape[-1])

# ============================================================
# EVALUATION
# ============================================================

total_images = 0
correct_predictions = 0

class_results = {}

print("\n" + "=" * 70)
print("PLANTDOC REAL-WORLD EVALUATION")
print("=" * 70)

for plantdoc_class in sorted(os.listdir(PLANTDOC_PATH)):

    class_folder = os.path.join(PLANTDOC_PATH, plantdoc_class)

    if not os.path.isdir(class_folder):
        continue

    # Skip classes that we haven't mapped
    if plantdoc_class not in PLANTDOC_MAPPING:
        print(f"\nSkipping: {plantdoc_class}")
        continue

    expected_class = PLANTDOC_MAPPING[plantdoc_class]

    expected_index = CLASS_NAMES.index(expected_class)

    class_total = 0
    class_correct = 0

    for filename in os.listdir(class_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            continue

        image_path = os.path.join(class_folder, filename)

        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(
                image_array,
                verbose=0
            )

            predicted_index = np.argmax(prediction[0])

            class_total += 1
            total_images += 1

            if predicted_index == expected_index:
                class_correct += 1
                correct_predictions += 1

        except Exception as e:
            print("Error:", image_path)
            print(e)

    class_accuracy = (
        class_correct / class_total * 100
        if class_total > 0 else 0
    )

    class_results[plantdoc_class] = (
        class_correct,
        class_total,
        class_accuracy
    )

    print(
        f"{plantdoc_class:<35} "
        f"{class_correct}/{class_total} "
        f"({class_accuracy:.2f}%)"
    )

# ============================================================
# FINAL RESULT
# ============================================================

overall_accuracy = (
    correct_predictions / total_images * 100
    if total_images > 0 else 0
)

print("\n" + "=" * 70)
print("FINAL PLANTDOC RESULT")
print("=" * 70)

print("Images evaluated :", total_images)
print("Correct          :", correct_predictions)
print("Incorrect        :", total_images - correct_predictions)
print(f"Accuracy         : {overall_accuracy:.2f}%")

print("=" * 70)