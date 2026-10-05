"""Classification metrics computed from a confusion matrix.

Deliberately numpy-only. scikit-learn is not installed in this project and
adding it just for four aggregate numbers would pull a large dependency into
a CPU-only TensorFlow environment.

Macro averages skip zero-support classes. That matters for the PlantDoc test
split: config.PLANTDOC_MAPPING covers 28 PlantVillage classes, but
"Tomato two spotted spider mites leaf" exists only in the PlantDoc train
folder, so only 27 of the 28 ever appear as labels at evaluation time. Averaging
over all 28 would fold in a class with no test images and depress the score for
a reason that has nothing to do with the model.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    """Return (num_classes, num_classes) int64 matrix.

    Entry [i, j] counts samples whose true label is i and whose prediction is j.
    Rows and columns outside 0..num_classes-1 are a class-ordering bug, so they
    raise rather than silently reshaping.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if values.size and (values.min() < 0 or values.max() >= num_classes):
            raise ValueError(
                f"{name} out of range [0, {num_classes}): "
                f"min={values.min()} max={values.max()}. "
                "A label outside this range means the class list does not match "
                "the model's output order."
            )

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    np.add.at(matrix, (y_true, y_pred), 1)

    return matrix


def num_classes_of(matrix):
    return np.asarray(matrix).shape[0]


def per_class_scores(matrix):
    """Return dicts of per-class precision/recall/F1/support.

    precision = TP / (TP + FP) -> how many predicted-i were really i
    recall    = TP / (TP + FN) -> how many real-i were found
    f1        = harmonic mean, defined as 0.0 when the denominator is 0

    A class with no predictions and no true samples gets support 0 and f1 0.0,
    but it is excluded from every macro average.
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)

    # Compute under explicit masks rather than np.where: np.where evaluates
    # both branches, so the division runs on zero denominators and emits
    # RuntimeWarnings even where the value is discarded.
    precision = np.zeros(num_classes_of(matrix), dtype=np.float64)
    recall = np.zeros_like(precision)

    has_predictions = predicted > 0
    has_samples = actual > 0

    precision[has_predictions] = true_positive[has_predictions] / predicted[has_predictions]
    recall[has_samples] = true_positive[has_samples] / actual[has_samples]

    denominator = precision + recall
    f1 = np.zeros_like(precision)
    scorable = denominator > 0
    f1[scorable] = (
        2.0 * precision[scorable] * recall[scorable] / denominator[scorable]
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual.astype(np.int64),
    }


def summarize(y_true, y_pred, num_classes, class_names=None, class_indices=None):
    """Accuracy plus macro and weighted precision/recall/F1.

    class_indices restricts the averaging set. Defaults to every class with
    non-zero support. Pass it explicitly when a dataset covers a subset of the
    label space, so the reported macro average has a stated denominator.
    """
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    scores = per_class_scores(matrix)

    support = scores["support"]
    total = int(support.sum())

    if total == 0:
        raise ValueError("No samples to score")

    correct = int(np.trace(matrix))
    accuracy = correct / total

    if class_indices is None:
        class_indices = np.flatnonzero(support > 0)

    class_indices = np.asarray(class_indices, dtype=np.int64)
    averaged = support[class_indices].sum()

    if averaged != total:
        raise ValueError(
            f"class_indices covers {averaged} samples but the matrix has {total}. "
            "Every true label must be inside the averaging set."
        )

    weights = support[class_indices] / total

    macro_precision = float(scores["precision"][class_indices].mean())
    macro_recall = float(scores["recall"][class_indices].mean())
    macro_f1 = float(scores["f1"][class_indices].mean())

    weighted_precision = float(np.dot(scores["precision"][class_indices], weights))
    weighted_recall = float(np.dot(scores["recall"][class_indices], weights))
    weighted_f1 = float(np.dot(scores["f1"][class_indices], weights))

    present = [
        {
            "class_index": int(i),
            "class_name": class_names[i] if class_names else str(i),
            "precision": float(scores["precision"][i]),
            "recall": float(scores["recall"][i]),
            "f1": float(scores["f1"][i]),
            "support": int(support[i]),
        }
        for i in class_indices
    ]

    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_averaged_over": int(len(class_indices)),
        "per_class": present,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def format_table(rows):
    """Console table. Percentages to 2dp, kept in step with the printed header."""
    header = (
        f"{'Model':<16}{'Accuracy':>10}{'Prec':>9}{'Recall':>9}{'F1':>9}"
        f"{'Correct':>9}{'Total':>8}"
    )

    lines = [header, "-" * len(header)]

    for row in rows:
        lines.append(
            f"{row['label']:<16}"
            f"{row['accuracy_percent']:>9.2f}%"
            f"{row['macro_precision'] * 100:>8.2f}%"
            f"{row['macro_recall'] * 100:>8.2f}%"
            f"{row['macro_f1'] * 100:>8.2f}%"
            f"{row['correct']:>9}{row['total']:>8}"
        )

    return "\n".join(lines)