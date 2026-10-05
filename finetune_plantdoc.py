import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import config
import plantdoc_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune PlantVillage checkpoints on a mixed "
                    "PlantVillage + PlantDoc training set."
    )

    parser.add_argument(
        "--model",
        choices=sorted(config.BASELINE_MODELS),
        default="resnet18_improved",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--plantvillage-per-class",
        type=int,
        default=200,
        help="Cap on PlantVillage train images sampled per class.",
    )
    parser.add_argument(
        "--plantdoc-per-class",
        type=int,
        default=None,
        help="Cap on PlantDoc train images per class. Default uses all.",
    )
    parser.add_argument(
        "--plantdoc-weight",
        type=float,
        default=2.0,
        help="Oversampling factor applied to the PlantDoc portion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save. Default: <model name>_plantdoc.keras",
    )
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=30,
        help="PlantVillage validation images per class.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)

    return parser.parse_args()


def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(height_factor=0.15, width_factor=0.15),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
            layers.RandomContrast(0.15),
        ],
        name="finetune_augmentation",
    )


def prepare_model(model_name, learning_rate):
    path = config.BASELINE_MODELS[model_name]

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"Loading {model_name} from {path.name}...")

    model = tf.keras.models.load_model(path)

    outputs = model.output_shape[-1]

    if outputs != config.NUM_CLASSES:
        raise ValueError(
            f"{path.name} outputs {outputs} classes, expected {config.NUM_CLASSES}"
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_on_plantdoc(model, batch_size):
    """PlantDoc test accuracy. Test split is never used for training."""
    normalization = layers.Rescaling(1.0 / 255)

    def preprocess_val(images, labels):
        return normalization(tf.cast(images, tf.float32)), labels

    dataset, _, _, _ = plantdoc_dataset.build_plantdoc_dataset(
        config.PLANTDOC_TEST_DIR,
        shuffle=False,
        seed=config.SEED,
        batch_size=batch_size,
    )

    dataset = dataset.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

    correct = 0
    seen = 0

    for images, batch_labels in dataset:
        predictions = model.predict(images, verbose=0)
        predicted = np.argmax(predictions, axis=1)

        correct += int(np.sum(predicted == batch_labels.numpy()))
        seen += len(batch_labels)

    return correct / seen * 100 if seen else 0.0


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    config.validate_split_alignment()

    output_path = args.output or (
        config.MODEL_DIR / f"{args.model}_plantdoc.keras"
    )

    print("=" * 70)
    print(f"Fine-tuning {args.model}")
    print("=" * 70)
    print(f"Epochs            : {args.epochs}")
    print(f"Learning rate     : {args.learning_rate}")
    print(f"PlantVillage/class: {args.plantvillage_per_class}")
    print(f"PlantDoc/class    : {args.plantdoc_per_class or 'all'}")
    print(f"PlantDoc weight   : {args.plantdoc_weight}")
    print(f"Output            : {output_path.name}")
    print()

    augmentation = build_augmentation()
    normalization = layers.Rescaling(1.0 / 255)

    def preprocess_train(images, labels):
        images = tf.cast(images, tf.float32)
        images = augmentation(images, training=True)
        return normalization(images), labels

    def preprocess_val(images, labels):
        images = tf.cast(images, tf.float32)
        return normalization(images), labels

    # ---------------------------------------------------------------
    # Training data: PlantVillage + PlantDoc mixed
    # ---------------------------------------------------------------

    pv_records = plantdoc_dataset.plantvillage_records(
        config.PLANTVILLAGE_TRAIN_DIR,
        per_class_cap=args.plantvillage_per_class,
        seed=args.seed,
    )

    pd_records = plantdoc_dataset.plantdoc_records(
        config.PLANTDOC_TRAIN_DIR,
        per_class_cap=args.plantdoc_per_class,
        seed=args.seed,
    )

    # Oversample PlantDoc so it carries real weight in each batch. It is only
    # ~6% of the raw mix, which is too little to shift the domain.
    pd_oversampled = pd_records * max(1, int(args.plantdoc_weight))

    records = pv_records + pd_oversampled

    random.Random(args.seed).shuffle(records)

    print(f"PlantVillage images : {len(pv_records)}")
    print(f"PlantDoc images     : {len(pd_records)} "
          f"(x{max(1, int(args.plantdoc_weight))} = {len(pd_oversampled)})")
    print(f"Total training      : {len(records)}")
    print()

    train_ds = plantdoc_dataset.build_path_dataset(
        records,
        shuffle=True,
        seed=args.seed,
        batch_size=args.batch_size,
    ).map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)

    # ---------------------------------------------------------------
    # Validation data: PlantVillage validation only
    # ---------------------------------------------------------------

    val_records = plantdoc_dataset.plantvillage_records(
        config.PLANTVILLAGE_VAL_DIR,
        per_class_cap=args.val_per_class,
        seed=args.seed,
    )

    print(f"Validation images   : {len(val_records)} "
          f"(PlantVillage, {args.val_per_class}/class)")
    print()

    val_ds = plantdoc_dataset.build_path_dataset(
        val_records,
        shuffle=False,
        seed=args.seed,
        batch_size=args.batch_size,
    ).map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

    model = prepare_model(args.model, args.learning_rate)

    baseline_plantdoc = evaluate_on_plantdoc(model, args.batch_size)
    print(f"Baseline PlantDoc test accuracy: {baseline_plantdoc:.2f}%")
    print()

    checkpoints_dir = config.MODEL_DIR / f"{args.model}_plantdoc_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoints_dir / "best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    started = time.time()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    elapsed = (time.time() - started) / 60

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    best_val = max(history.history["val_accuracy"])

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best validation accuracy (PlantVillage) : {best_val * 100:.2f}%")
    print(f"Baseline PlantDoc test accuracy         : {baseline_plantdoc:.2f}%")

    final_plantdoc = evaluate_on_plantdoc(model, args.batch_size)

    print(f"Fine-tuned PlantDoc test accuracy       : {final_plantdoc:.2f}%")
    print(f"Change                                  : "
          f"{final_plantdoc - baseline_plantdoc:+.2f} pts")
    print(f"Training time                           : {elapsed:.1f} min")
    print(f"Saved to                                : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)