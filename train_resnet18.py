import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

TRAIN_DIR = os.path.join(BASE_DIR, "split", "train")
VAL_DIR = os.path.join(BASE_DIR, "split", "validation")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet18_plant_disease.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38
EPOCHS = 10

# ============================================================
# CHECK FOLDERS
# ============================================================

print("Checking dataset folders...")

print("Train folder:", TRAIN_DIR)
print("Validation folder:", VAL_DIR)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError("Training folder not found!")

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError("Validation folder not found!")

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

class_names = train_ds.class_names

print("\nClasses found:", len(class_names))

for i, name in enumerate(class_names):
    print(i, name)

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}"
    )

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

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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

def build_resnet18():

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

    # Final 38-class classifier
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet18_PlantDisease"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

print("\nBuilding ResNet18...")

model = build_resnet18()

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
print("STARTING RESNET18 TRAINING")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RESNET18 TRAINING COMPLETE")
print("=" * 60)

print(
    "Best Validation Accuracy:",
    max(history.history["val_accuracy"]) * 100
)

print(
    "Best Training Accuracy:",
    max(history.history["accuracy"]) * 100
)

print("\nModel saved at:")
print(MODEL_PATH)

print("=" * 60)