import os
import tensorflow as tf

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

TEST_DIR = os.path.join(
    BASE_DIR,
    "split",
    "test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "mobilenetv2_plant_disease.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================================================
# LOAD TEST DATA
# ============================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ============================================================
# NORMALIZATION
# ============================================================

test_ds = test_ds.map(
    lambda images, labels: (
        tf.cast(images, tf.float32) / 255.0,
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading MobileNetV2...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")

# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating MobileNetV2 on PlantVillage test set...\n")

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)

# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("MOBILENETV2 - PLANTVILLAGE TEST RESULT")
print("=" * 60)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

print("=" * 60)