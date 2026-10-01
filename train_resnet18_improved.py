import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = r"C:\Plant_disease_detection"

TRAIN_DIR = os.path.join(BASE_DIR, "split", "train")
VAL_DIR = os.path.join(BASE_DIR, "split", "validation")

# IMPORTANT: save as a new model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "resnet18_improved.keras"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38

# Stage 1 + Stage 2
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

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
# DATA AUGMENTATION
# ============================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.15
        ),

        layers.RandomZoom(
            height_factor=0.15,
            width_factor=0.15
        ),

        layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomContrast(
            0.15
        ),

    ],
    name="plant_augmentation"
)

# ============================================================
# NORMALIZATION
# ============================================================

normalization = layers.Rescaling(
    1.0 / 255
)

# ============================================================
# PREPROCESS DATASET
# ============================================================

def preprocess_train(images, labels):

    images = tf.cast(
        images,
        tf.float32
    )

    images = augmentation(
        images,
        training=True
    )

    images = normalization(
        images
    )

    return images, labels


def preprocess_validation(images, labels):

    images = tf.cast(
        images,
        tf.float32
    )

    images = normalization(
        images
    )

    return images, labels


train_ds = train_ds.map(
    preprocess_train,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_validation,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(
    tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    tf.data.AUTOTUNE
)

# ============================================================
# RESNET18 BASIC BLOCK
# ============================================================

def residual_block(
    x,
    filters,
    stride=1
):

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

    # Adjust shortcut
    if (
        stride != 1
        or shortcut.shape[-1] != filters
    ):

        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # Residual connection
    x = layers.Add()(
        [x, shortcut]
    )

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

    # Stage 1
    x = residual_block(
        x,
        64,
        stride=1
    )

    x = residual_block(
        x,
        64,
        stride=1
    )

    # Stage 2
    x = residual_block(
        x,
        128,
        stride=2
    )

    x = residual_block(
        x,
        128,
        stride=1
    )

    # Stage 3
    x = residual_block(
        x,
        256,
        stride=2
    )

    x = residual_block(
        x,
        256,
        stride=1
    )

    # Stage 4
    x = residual_block(
        x,
        512,
        stride=2
    )

    x = residual_block(
        x,
        512,
        stride=1
    )

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout to reduce overfitting
    x = layers.Dropout(
        0.30
    )(x)

    # 38-class classifier
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="ResNet18_PlantDisease_Improved"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

print("\nBuilding improved ResNet18...")

model = build_resnet18()

model.summary()

# ============================================================
# COMPILE - STAGE 1
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
# STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 1: AUGMENTED RESNET18 TRAINING")
print("=" * 60)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================================
# STAGE 2 - LOWER LEARNING RATE FINE-TUNING
# ============================================================

print("\n")
print("=" * 60)
print("STAGE 2: FINE-TUNING")
print("=" * 60)

# Recompile with smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================================
# FINAL RESULTS
# ============================================================

best_val_accuracy = max(
    history1.history["val_accuracy"]
    + history2.history["val_accuracy"]
)

best_train_accuracy = max(
    history1.history["accuracy"]
    + history2.history["accuracy"]
)

print("\n")
print("=" * 60)
print("IMPROVED RESNET18 TRAINING COMPLETE")
print("=" * 60)

print(
    "Best Validation Accuracy:",
    best_val_accuracy * 100
)

print(
    "Best Training Accuracy:",
    best_train_accuracy * 100
)

print("\nModel saved at:")
print(MODEL_PATH)

print("=" * 60)