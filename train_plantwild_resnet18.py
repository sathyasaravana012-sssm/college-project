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
    "plantwild_resnet18.keras"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plantwild_resnet18_results.json"
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
# CHECK FOLDERS
# ============================================================

print("Checking dataset folders...")

print("Train folder:", TRAIN_DIR)
print("Validation folder:", VAL_DIR)
print("Test folder:", TEST_DIR)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError("Training folder not found!")

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError("Validation folder not found!")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError("Test folder not found!")

# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names

print("\nClasses found:", len(class_names))

for i, name in enumerate(class_names):
    print(i, name)

if len(class_names) != EXPECTED_CLASSES:
    raise ValueError(
        f"Expected {EXPECTED_CLASSES} classes, "
        f"but found {len(class_names)}"
    )

if val_ds.class_names != class_names or test_ds.class_names != class_names:
    raise ValueError("Class folders do not match across splits!")

# ============================================================
# NORMALIZATION
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
# RESNET18 BASIC BLOCK
# ============================================================

def residual_block(x, filters, stride=1):

    shortcut = x

    # First convolution
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions change
    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # Residual connection
    x = layers.Add()([x, shortcut])

    x = layers.ReLU()(x)

    return x


# ============================================================
# BUILD RESNET18
# ============================================================

def build_resnet18(num_classes):

    inputs = layers.Input(
        shape=(224, 224, 3)
    )

    # Initial convolution
    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        padding="same",
        use_bias=False
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same"
    )(x)

    # --------------------------------------------------------
    # Stage 1
    # 64 filters × 2 blocks
    # --------------------------------------------------------

    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)

    # --------------------------------------------------------
    # Stage 2
    # 128 filters × 2 blocks
    # --------------------------------------------------------

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)

    # --------------------------------------------------------
    # Stage 3
    # 256 filters × 2 blocks
    # --------------------------------------------------------

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)

    # --------------------------------------------------------
    # Stage 4
    # 512 filters × 2 blocks
    # --------------------------------------------------------

    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier head
    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet18_PlantWild"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

print("\nBuilding ResNet18...")

model = build_resnet18(EXPECTED_CLASSES)

model.summary()

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

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING RESNET18 TRAINING (PLANTWILD)")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# TEST EVALUATION
# ============================================================

print("\nLoading best saved model...")
best_model = tf.keras.models.load_model(MODEL_PATH)

print("\n")
print("=" * 60)
print("EVALUATING ON PLANTWILD TEST SET")
print("=" * 60)

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
print("FINAL RESNET18 RESULTS (PLANTWILD)")
print("=" * 60)

print(f"Model             : ResNet18 (PlantWild)")
print(f"Best Val Accuracy : {max(history.history['val_accuracy']) * 100:.2f}%")
print(f"Test Accuracy     : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {precision * 100:.2f}%")
print(f"Macro Recall      : {recall * 100:.2f}%")
print(f"Macro F1-score    : {f1 * 100:.2f}%")
print(f"Correct           : {int(np.sum(y_true == y_pred))}")
print(f"Total             : {len(y_true)}")
print("=" * 60)

results = {
    "model": "ResNet18",
    "dataset": "PlantWild (34 classes)",
    "architecture": "ResNet-18 (Keras, trained from scratch)",
    "image_size": [224, 224],
    "num_classes": EXPECTED_CLASSES,
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

print("\nModel saved at:")
print(MODEL_PATH)

print("\nResults saved at:")
print(RESULTS_PATH)

print("=" * 60)
