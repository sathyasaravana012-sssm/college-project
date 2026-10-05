import os
from pathlib import Path

import tensorflow as tf

import config


def _central_crop_resize(image):
    target_h, target_w = config.IMAGE_SIZE
    shape = tf.shape(image)

    image_h, image_w = shape[0], shape[1]
    target = tf.minimum(image_h, image_w)

    padded_center_crop = tf.image.resize_with_crop_or_pad(
        image,
        target,
        target,
    )

    resized = tf.image.resize(padded_center_crop, config.IMAGE_SIZE)

    return tf.cast(resized, tf.float32)


def _decode(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)

    # Central-crop to square then resize, so no part of the leaf is stretched.
    # Kept in [0, 255] float to match image_dataset_from_directory, which yields
    # uint8. Augmentation (RandomContrast) and /255 rescaling are applied later
    # in the shared preprocessing step.
    image = _central_crop_resize(image)

    return image, label


def build_path_dataset(records, shuffle, seed, batch_size=config.BATCH_SIZE):
    """records: iterable of (path_string, int_label)."""
    records = list(records)

    if not records:
        raise ValueError("No records supplied")

    paths_tensor = tf.constant([r[0] for r in records])
    labels_tensor = tf.constant([r[1] for r in records], dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
    dataset = dataset.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(
            len(records),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plantvillage_records(directory, per_class_cap=None, seed=config.SEED):
    import random

    rng = random.Random(seed)

    records = []

    for folder in config.class_folders(directory):
        label = config.CLASS_TO_INDEX[folder]
        files = config.image_files(Path(directory) / folder)

        if per_class_cap is not None and len(files) > per_class_cap:
            files = rng.sample(files, per_class_cap)

        for name in files:
            records.append((str(Path(directory) / folder / name), label))

    return records


def plantdoc_records(directory, per_class_cap=None, seed=config.SEED):
    import random

    rng = random.Random(seed)

    records = []

    for folder in config.class_folders(directory):
        if folder not in config.PLANTDOC_TO_INDEX:
            raise ValueError(f"Unmapped PlantDoc folder: {folder}")

        label = config.PLANTDOC_TO_INDEX[folder]
        files = config.image_files(Path(directory) / folder)

        if per_class_cap is not None and len(files) > per_class_cap:
            files = rng.sample(files, per_class_cap)

        for name in files:
            records.append((str(Path(directory) / folder / name), label))

    return records


def build_mixed_training_set(
    plantdoc_per_class_cap=None,
    plantvillage_per_class_cap=200,
    seed=config.SEED,
):
    pv = plantvillage_records(
        config.PLANTVILLAGE_TRAIN_DIR,
        per_class_cap=plantvillage_per_class_cap,
        seed=seed,
    )

    pd = plantdoc_records(
        config.PLANTDOC_TRAIN_DIR,
        per_class_cap=plantdoc_per_class_cap,
        seed=seed,
    )

    return pv + pd, len(pv), len(pd)


def build_plantdoc_dataset(directory, shuffle, seed, batch_size=config.BATCH_SIZE):
    """Returns (dataset, folder_names, paths, labels).

    folder_names / paths / labels are aligned element-for-element with each
    other so per-class breakdowns can never desync from the batches.
    """
    directory = Path(directory)

    folders = config.class_folders(directory)

    unmapped = [f for f in folders if f not in config.PLANTDOC_TO_INDEX]

    if unmapped:
        raise ValueError(
            f"Unmapped PlantDoc folders in {directory}:\n  {unmapped}\n"
            "Add them to config.PLANTDOC_MAPPING before training or evaluating."
        )

    paths = []
    labels = []

    for folder in folders:
        label = config.PLANTDOC_TO_INDEX[folder]

        for name in config.image_files(directory / folder):
            paths.append(str(directory / folder / name))
            labels.append(label)

    if not paths:
        raise ValueError(f"No images found in {directory}")

    paths_tensor = tf.constant(paths)
    labels_tensor = tf.constant(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    dataset = dataset.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)

    return dataset, folders, paths, labels


def build_plantvillage_dataset(directory, shuffle, seed, batch_size=config.BATCH_SIZE):
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        image_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="int",
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )

    return dataset


def assert_dataset_class_order(dataset):
    names = list(dataset.class_names)

    if names != config.CLASS_NAMES:
        raise ValueError(
            "Dataset class order does not match config.CLASS_NAMES.\n"
            f"dataset: {names}\nconfig:  {config.CLASS_NAMES}"
        )

    return names