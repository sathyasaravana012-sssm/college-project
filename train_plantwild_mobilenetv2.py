import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

DEFAULT_TRAIN_DIR = os.path.join(BASE_DIR, "plantwild", "split", "train")
DEFAULT_VAL_DIR   = os.path.join(BASE_DIR, "plantwild", "split", "validation")
DEFAULT_TEST_DIR  = os.path.join(BASE_DIR, "plantwild", "split", "test")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plantwild_mobilenetv2.keras"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plantwild_mobilenetv2_results.json"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EXPECTED_CLASSES = 34

parser = argparse.ArgumentParser()
parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR)
parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR)
parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

TRAIN_DIR = args.train_dir
VAL_DIR   = args.val_dir
TEST_DIR  = args.test_dir
EPOCHS    = args.epochs

# ============================================================
# LOAD DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nClasses found:", NUM_CLASSES)

for i, name in enumerate(class_names):
    print(i, name)

if NUM_CLASSES != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, but found {NUM_CLASSES}."
    )

if val_ds.class_names != class_names or test_ds.class_names != class_names:
    raise ValueError("Class folders do not match across splits!")

# ============================================================
# PREPROCESSING
# ============================================================

normalization = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (normalization(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.05, 0.05),
], name="augmentation")

# ============================================================
# MOBILENETV2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting MobileNetV2 training (PlantWild)...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\nLoading best saved model...")
best_model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating on PlantWild test set...")

test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = best_model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("\n")
print("=" * 60)
print("FINAL MOBILENETV2 RESULTS (PLANTWILD)")
print("=" * 60)

print(f"Model             : MobileNetV2 (PlantWild)")
print(f"Best Val Accuracy : {max(history.history['val_accuracy']) * 100:.2f}%")
print(f"Test Accuracy     : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision * 100:.2f}%")
print(f"Macro Recall      : {recall * 100:.2f}%")
print(f"Macro F1-score    : {f1 * 100:.2f}%")
print(f"Correct           : {int(np.sum(y_true == y_pred))}")
print(f"Total             : {len(y_true)}")
print("=" * 60)

results = {
    "model": "MobileNetV2",
    "dataset": "PlantWild (34 classes)",
    "architecture": "MobileNetV2 (ImageNet weights, frozen backbone)",
    "image_size": [224, 224],
    "num_classes": NUM_CLASSES,
    "epochs_requested": EPOCHS,
    "best_val_accuracy": float(max(history.history["val_accuracy"])),
    "test_loss": float(test_loss),
    "test_accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "correct": int(np.sum(y_true == y_pred)),
    "total": int(len(y_true)),
    "class_names": class_names,
    "model_path": MODEL_PATH
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nTraining completed.")

print("\nBest model saved at:")
print(MODEL_PATH)

print("\nResults saved at:")
print(RESULTS_PATH)
