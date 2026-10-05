"""Final evaluation of the four reported model rows, on both datasets.

Reports accuracy, macro precision, macro recall and macro F1 for:
    1. Simple CNN    (plant_disease_model.keras)
    2. ResNet18      (resnet18_improved.keras)
    3. MobileNetV2   (mobilenetv2_plant_disease.keras)
    4. Ensemble      (unweighted mean of the three softmax outputs)

On PlantVillage test (split/test, 38 classes, all present) and on the PlantDoc
real-world test set.

Two things this script is careful about, both learned from the earlier
evaluation scripts in this repo:

Class names come from config.CLASS_NAMES, which is derived from disk at import
time and checked to contain exactly NUM_CLASSES entries. Nine older scripts
hardcode their own list and several are stale -- evaluate_plantdoc.py still
says "Cherry___Powdery_mildew" where the directory is
"Cherry_(including_sour)___Powdery_mildew". Those scripts report accuracy
against a class order that does not match the models.

Models are called through predict(), which runs in inference mode.
mobilenetv2_plant_disease.keras carries its augmentation inside the graph as
layer 2; in training mode that layer would randomly flip and rotate every test
image and produce numbers that are just noise. Checked directly: predict(x)
matches model(x, training=False) to 1e-6 and does not match
model(x, training=True).

PlantDoc macro averages cover only the classes actually present in the test
folder. config.PLANTDOC_MAPPING maps 28 PlantVillage classes, but
"Tomato two spotted spider mites leaf" exists only in the PlantDoc train
folder, so 27 of the 28 appear as labels at test time.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

import config
import metrics
import plantdoc_dataset

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The three single-model rows, in report order. "basic" is a MobileNetV2
# backbone with augmentation in the data pipeline; "mobilenetv2" is the same
# backbone with augmentation in-graph. They are listed separately because the
# accepted abstract names both.
ENSEMBLE_MEMBERS = (
    ("Simple CNN", "basic"),
    ("ResNet18", "resnet18_improved"),
    ("MobileNetV2", "mobilenetv2"),
)

# Recorded by the fine-tuning runs; see README.md and logs_ft_basic.txt.
# Reproduced here verbatim so the report and the training log agree.
FINE_TUNING_RECORD = {
    "model": "basic_plantdoc",
    "plantdoc_test": {
        "baseline_accuracy_percent": 26.27,
        "finetuned_accuracy_percent": 28.39,
        "improvement_points": 2.12,
    },
    "source": "logs_ft_basic.txt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the four reported model rows on PlantVillage "
                    "and PlantDoc."
    )

    parser.add_argument(
        "--datasets",
        choices=["plantvillage", "plantdoc", "both"],
        default="both",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys from config.BASELINE_MODELS, or 'all'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N images per dataset. For a fast "
             "smoke test; the report must be produced without this.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "final_evaluations.json",
    )

    return parser.parse_args()


def load_predictions(model_name, images, batch_size, cache=None):
    """Softmax output for every image, one model load per checkpoint.

    Returns a (len(images), NUM_CLASSES) float32 array.

    cache is scoped to a single dataset by the caller. It holds 8176 x 38
    float32 (~1.2 MB) per model, so there is no reason to share it across
    datasets -- and sharing would be a bug, since the arrays correspond to
    different images.
    """
    if cache is not None and model_name in cache:
        return cache[model_name]

    path = config.BASELINE_MODELS[model_name]

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = tf.keras.models.load_model(path)

    outputs = model.output_shape[-1]
    if outputs != config.NUM_CLASSES:
        raise ValueError(
            f"{path.name} outputs {outputs} classes, expected {config.NUM_CLASSES}"
        )

    batches = []
    started = time.time()

    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]

        # predict() runs the model in inference mode, which is what keeps the
        # in-graph augmentation layer a no-op. Verified: predict(x) equals
        # model(x, training=False) to within 1e-6 and does NOT equal
        # model(x, training=True). Keras 3 rejects an explicit training= kwarg
        # on predict(), so inference mode here is implicit.
        batches.append(model.predict(chunk, verbose=0))

    probabilities = np.concatenate(batches, axis=0).astype(np.float32)
    elapsed = time.time() - started

    print(f"    {model_name:<26} {len(images)} images in {elapsed:7.1f}s")

    del model

    if cache is not None:
        cache[model_name] = probabilities

    return probabilities


def gather_plantdoc(batch_size, limit=None):
    """PlantDoc test images as float32 (N, 224, 224, 3) in [0, 1], plus labels.

    Uses plantdoc_dataset.build_plantdoc_dataset so the central-crop matches
    training exactly, then flattens the batches. Holds all 236 images in memory
    (about 142 MB), which is fine at this scale and avoids a second decode pass.
    """
    dataset, folders, paths, labels = plantdoc_dataset.build_plantdoc_dataset(
        config.PLANTDOC_TEST_DIR,
        shuffle=False,
        seed=config.SEED,
        batch_size=batch_size,
    )

    images = np.concatenate(
        [batch_images.numpy() for batch_images, _ in dataset], axis=0
    )
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset])

    images = images.astype(np.float32) / 255.0

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    print(f"  PlantDoc: {len(images)} images, {len(folders)} folders on disk")

    return images, labels.astype(np.int64)


def gather_plantvillage(batch_size, limit=None):
    """PlantVillage test images and labels, label_mode='int'.

    image_dataset_from_directory assigns integer labels from sorted(os.listdir),
    which is the order config.CLASS_NAMES reproduces, so the two agree by
    construction here rather than by two hardcoded lists matching.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(config.PLANTVILLAGE_TEST_DIR),
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    if dataset.class_names != config.CLASS_NAMES:
        raise ValueError(
            "PlantVillage test class order does not match config.CLASS_NAMES.\n"
            f"dataset: {dataset.class_names}\n"
            f"config : {config.CLASS_NAMES}"
        )

    images = np.concatenate(
        [batch_images.numpy() for batch_images, _ in dataset], axis=0
    )
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset])

    images = images.astype(np.float32) / 255.0

    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    print(f"  PlantVillage: {len(images)} images, {len(dataset.class_names)} classes")

    return images, labels.astype(np.int64)


def evaluate_dataset(name, images, labels, model_keys, batch_size, cache):
    """Run every requested model plus the ensemble over one dataset.

    cache is populated here and used only by this call's ensemble step. The
    caller passes a fresh dict per dataset.
    """
    print(f"\n{'=' * 70}")
    print(name)
    print("=" * 70)

    # Only classes with real test images enter the macro averages.
    present = np.flatnonzero(np.bincount(labels, minlength=config.NUM_CLASSES) > 0)

    print(f"  Classes with test images: {len(present)} of {config.NUM_CLASSES}")
    print(f"  Macro metrics averaged over {len(present)} classes")
    print()

    predicted_labels = {}

    for key in model_keys:
        print(f"  Predicting with {key}...")
        probabilities = load_predictions(key, images, batch_size, cache)
        predicted_labels[key] = np.argmax(probabilities, axis=1)

    rows = []
    results = {}

    for label, key in ENSEMBLE_MEMBERS:
        if key not in predicted_labels:
            continue

        results[label] = metrics.summarize(
            labels,
            predicted_labels[key],
            config.NUM_CLASSES,
            class_names=config.CLASS_NAMES,
            class_indices=present,
        )
        results[label]["model_key"] = key
        results[label]["checkpoint"] = config.BASELINE_MODELS[key].name
        rows.append({"label": label, **results[label]})

    member_keys = [key for _, key in ENSEMBLE_MEMBERS if key in predicted_labels]

    if len(member_keys) > 1:
        # Unweighted soft voting: mean the softmax vectors, then argmax. No
        # weights are fitted, so nothing here needs a held-out split and the
        # result is deterministic.
        mean_probabilities = np.stack(
            [cache[key] for key in member_keys]
        ).mean(axis=0)
        ensemble_predictions = np.argmax(mean_probabilities, axis=1)

        results["Ensemble"] = metrics.summarize(
            labels,
            ensemble_predictions,
            config.NUM_CLASSES,
            class_names=config.CLASS_NAMES,
            class_indices=present,
        )
        results["Ensemble"]["model_key"] = "+".join(member_keys)
        results["Ensemble"]["checkpoint"] = "mean softmax"
        rows.append({"label": "Ensemble", **results["Ensemble"]})

    print()
    print(metrics.format_table(rows))
    print()

    best_single = max(
        (row for row in rows if row["label"] != "Ensemble"),
        key=lambda row: row["accuracy_percent"],
        default=None,
    )

    if best_single and results.get("Ensemble"):
        delta = (
            results["Ensemble"]["accuracy_percent"]
            - best_single["accuracy_percent"]
        )
        print(
            f"  Ensemble {results['Ensemble']['accuracy_percent']:.2f}% vs "
            f"best single {best_single['label']} "
            f"{best_single['accuracy_percent']:.2f}%  ({delta:+.2f} pts)"
        )
        print()

    return {
        "images": int(len(images)),
        "classes_with_images": int(len(present)),
        "class_indices_present": [int(i) for i in present],
        "class_names_present": [config.CLASS_NAMES[i] for i in present],
        "results": results,
    }


def main():
    args = parse_args()

    if args.models == "all":
        model_keys = [key for _, key in ENSEMBLE_MEMBERS]
    else:
        model_keys = [k.strip() for k in args.models.split(",") if k.strip()]

    for key in model_keys:
        if key not in config.BASELINE_MODELS:
            raise SystemExit(
                f"Unknown model '{key}'. "
                f"Available: {sorted(config.BASELINE_MODELS)}"
            )

    config.validate_split_alignment()

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_method": "unweighted mean of softmax outputs",
        "fine_tuning_record": FINE_TUNING_RECORD,
        "datasets": {},
    }

    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)
    print(f"Models: {', '.join(model_keys)}")
    if args.limit is not None:
        print(f"WARNING: --limit {args.limit} set. Not a reportable run.")

    # A fresh cache per dataset: the arrays hold probabilities for that
    # dataset's images, so carrying one over would score the wrong pictures.
    if args.datasets in ("plantvillage", "both"):
        images, labels = gather_plantvillage(args.batch_size, args.limit)

        report["datasets"]["plantvillage_test"] = evaluate_dataset(
            "PlantVillage test (split/test)",
            images,
            labels,
            model_keys,
            args.batch_size,
            cache={},
        )
        del images

    if args.datasets in ("plantdoc", "both"):
        images, labels = gather_plantdoc(args.batch_size, args.limit)

        report["datasets"]["plantdoc_test"] = evaluate_dataset(
            "PlantDoc test (real-world)",
            images,
            labels,
            model_keys,
            args.batch_size,
            cache={},
        )
        del images

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()