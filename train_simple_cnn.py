import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = r"C:\Plant_disease_detection\split\train"
VAL_DIR   = r"C:\Plant_disease_detection\split\validation"
TEST_DIR  = r"C:\Plant_disease_detection\split\test"

MODEL_DIR = r"C:\Plant_disease_detection\model"
MODEL_PATH = os.path.join(MODEL_DIR, "simple_cnn_plant_disease.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 70)
print("SIMPLE CNN - PLANT DISEASE CLASSIFICATION")
print("=" * 70)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("\nNumber of classes:", NUM_CLASSES)
print("Classes:")

for i, name in enumerate(class_names):
    print(f"{i:2d}: {name}")


# ============================================================
# 3. CHECK THAT ALL DATASETS HAVE SAME CLASSES
# ============================================================

if val_ds.class_names != class_names:
    raise ValueError(
        "Validation classes do not match training classes!"
    )

if test_ds.class_names != class_names:
    raise ValueError(
        "Test classes do not match training classes!"
    )

if NUM_CLASSES != 38:
    raise ValueError(
        f"Expected 38 classes, but found {NUM_CLASSES}."
    )


# ============================================================
# 4. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# ============================================================
# 5. BUILD A GENUINE CNN FROM SCRATCH
# ============================================================

print("\nBuilding Simple CNN...")

model = models.Sequential([
    
    # Input
    layers.Input(shape=(224, 224, 3)),

    # Normalize pixels from 0-255 to 0-1
    layers.Rescaling(1.0 / 255),

    # -------- Block 1 --------
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------- Block 2 --------
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------- Block 3 --------
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # -------- Classification head --------
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 6. DISPLAY ARCHITECTURE
# ============================================================

print("\nMODEL ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. CALLBACKS
# ============================================================

callbacks = [

    # Save the best validation model
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    # Stop if validation performance stops improving
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate when validation performance stalls
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 9. TRAIN
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# 10. LOAD BEST MODEL
# ============================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# 11. EVALUATE ON TEST DATA
# ============================================================

print("\n")
print("=" * 70)
print("EVALUATING ON PLANTVILLAGE TEST SET")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# 12. GET PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = best_model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# 13. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

correct = int(np.sum(y_true == y_pred))
total = len(y_true)


# ============================================================
# 14. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL SIMPLE CNN RESULTS")
print("=" * 70)

print(f"Model             : Simple CNN")
print(f"Test Accuracy     : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision * 100:.2f}%")
print(f"Macro Recall      : {recall * 100:.2f}%")
print(f"Macro F1-score    : {f1 * 100:.2f}%")
print(f"Correct           : {correct}")
print(f"Total             : {total}")
print("=" * 70)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results = {
    "model": "Simple CNN",
    "architecture": "Custom CNN trained from scratch",
    "image_size": [224, 224],
    "num_classes": NUM_CLASSES,
    "epochs_requested": EPOCHS,
    "test_accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
    "correct": correct,
    "total": total,
    "model_path": MODEL_PATH
}

results_path = os.path.join(
    MODEL_DIR,
    "simple_cnn_results.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nResults saved to:")
print(results_path)

print("\nTraining and evaluation completed.")