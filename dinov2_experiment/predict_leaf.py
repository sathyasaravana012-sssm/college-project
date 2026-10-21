"""Predict a single leaf image with the trained DINOv2 pipeline.

    .venv\\Scripts\\python.exe predict_leaf.py "C:\\path\\to\\leaf.jpg"
    .venv\\Scripts\\python.exe predict_leaf.py leaf.jpg --top-k 5

Loads the frozen DINOv2 backbone (cached after the first download) and the
logistic-regression classifier saved by train_classifier.py, applies the exact
training-time preprocessing, and prints the predicted class plus the top-k
probabilities.

Set PDD_DATASET=plantwild to use the 34-class PlantWild classifier instead
of the default 38-class PlantVillage one.

A confidence score is just the classifier's probability estimate; it does NOT
guarantee the prediction is correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import dinov2_common as common


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to a leaf image.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backbone", default=common.BACKBONE_NAME)
    parser.add_argument("--crop", type=int, default=common.DINOV2_CROP)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import joblib

    classifier_path = common.artifact_path("dinov2_logreg.joblib")
    mapping_path = common.artifact_path("class_mapping.json")

    if not classifier_path.exists() or not mapping_path.exists():
        print("ERROR: trained classifier not found.")
        print(f"  expected: {classifier_path}")
        print(f"  expected: {mapping_path}")
        print("Run train_classifier.py first.")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    try:
        with Image.open(image_path) as img:
            tensor = common.preprocess_pil(img, resize=common.DINOV2_RESIZE, crop=args.crop)
    except Exception as exc:
        print(f"ERROR: could not read image {image_path}: {exc}")
        return 1

    mapping = common.load_json(mapping_path)
    class_names = mapping["class_names"]
    pipeline = joblib.load(classifier_path)
    backbone = common.load_backbone(device=args.device)

    features = common.extract_features_batch(backbone, tensor.unsqueeze(0), args.device)
    probs = pipeline.predict_proba(features)[0]

    order = np.argsort(probs)[::-1][:args.top_k]
    best = int(order[0])

    print("=" * 55)
    print("DINOv2 LEAF PREDICTION")
    print("=" * 55)
    print(f"Image      : {image_path}")
    print(f"Prediction : {class_names[best]}  (index {best})")
    print(f"Confidence : {probs[best]*100:.2f}%")
    print(f"Note       : confidence is an estimate, not a correctness guarantee.")
    print("-" * 55)
    print(f"Top {args.top_k}:")
    for idx in order:
        print(f"  {class_names[int(idx)]:<55} {probs[int(idx)]*100:6.2f}%")
    print("=" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
