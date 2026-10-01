import tensorflow as tf
import numpy as np
from pathlib import Path

# ==============================
# SETTINGS
# ==============================

MODEL_PATH = r"C:\Plant_disease_detection\model\resnet18_plant_disease.keras"
TEST_DIR = r"C:\Plant_disease_detection\split\test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# LOAD MODEL
# ==============================

print("Loading ResNet18 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ==============================
# LOAD TEST DATA
# ==============================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

print("\nNumber of classes:", len(class_names))
print("Classes:", class_names)

# ==============================
# PREDICTION
# ==============================

print("\nEvaluating ResNet18...")
print("Please wait...")

y_true = []
y_pred = []

for images, labels in test_ds:
    images = tf.cast(images, tf.float32) / 255.0

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ==============================
# RESULTS
# ==============================

correct = np.sum(y_true == y_pred)
total = len(y_true)
incorrect = total - correct

accuracy = correct / total * 100

print("\n" + "=" * 60)
print("RESNET18 PLANTVILLAGE TEST RESULT")
print("=" * 60)

print(f"Total images : {total}")
print(f"Correct      : {correct}")
print(f"Incorrect    : {incorrect}")
print(f"Accuracy     : {accuracy:.2f}%")

print("=" * 60)