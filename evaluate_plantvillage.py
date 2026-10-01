import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Plant_disease_detection\model\plant_disease_model.keras"
TEST_DIR = r"C:\Plant_disease_detection\split\test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print("Number of output classes:", model.output_shape[-1])

# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = sorted([
    folder for folder in os.listdir(TEST_DIR)
    if os.path.isdir(os.path.join(TEST_DIR, folder))
])

print("Number of classes:", len(class_names))

# ============================================================
# COLLECT IMAGES
# ============================================================

image_paths = []
true_labels = []

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

print("\nScanning test dataset...")

for label, class_name in enumerate(class_names):

    class_folder = os.path.join(TEST_DIR, class_name)

    for filename in os.listdir(class_folder):

        if filename.lower().endswith(valid_extensions):

            image_paths.append(
                os.path.join(class_folder, filename)
            )

            true_labels.append(label)

print("Total test images:", len(image_paths))

# ============================================================
# BATCH PREDICTION
# ============================================================

correct = 0
total = len(image_paths)

print("\nStarting evaluation...")
print("=" * 60)

for start in range(0, total, BATCH_SIZE):

    end = min(start + BATCH_SIZE, total)

    batch_paths = image_paths[start:end]
    batch_labels = true_labels[start:end]

    batch_images = []

    for path in batch_paths:

        try:

            image = Image.open(path).convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.array(image, dtype=np.float32)

            # Same normalization used during training
            image_array = image_array / 255.0

            batch_images.append(image_array)

        except Exception as e:

            print("\nError loading:", path)
            print(e)

    if len(batch_images) == 0:
        continue

    batch_images = np.array(batch_images)

    # Predict entire batch at once
    predictions = model.predict(
        batch_images,
        verbose=0
    )

    predicted_labels = np.argmax(predictions, axis=1)

    for predicted, actual in zip(predicted_labels, batch_labels):

        if predicted == actual:
            correct += 1

    # Progress
    processed = end

    accuracy = (correct / processed) * 100

    print(
        f"Processed {processed}/{total} "
        f"({processed / total * 100:.1f}%) "
        f"| Current Accuracy: {accuracy:.2f}%"
    )

# ============================================================
# FINAL RESULT
# ============================================================

final_accuracy = (correct / total) * 100

print("\n")
print("=" * 60)
print("           PLANTVILLAGE TEST RESULTS")
print("=" * 60)

print("Total Images :", total)
print("Correct      :", correct)
print("Incorrect    :", total - correct)

print(f"Accuracy     : {final_accuracy:.2f}%")

print("=" * 60)