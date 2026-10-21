"""Extract and cache frozen DINOv2 features for one or more splits.

Pipeline: leaf image -> RGB -> DINOv2 eval preprocessing -> frozen backbone
-> 384-d CLS feature vector. No augmentation and no fine-tuning.

Features are cached under ``dinov2_experiment/features`` and reused by
``train_classifier.py``, ``evaluate_models.py`` and ``predict_leaf.py``. The
run is resumable: it checkpoints every ``--save-every`` images, so an
interrupted CPU run continues where it stopped.

Examples
--------
Quick smoke test (500 images per class of the train split):
    .venv\\Scripts\\python.exe extract_features.py --splits train --limit-per-class 500

Full run (this is the long one; see README for a time estimate):
    .venv\\Scripts\\python.exe extract_features.py --batch-size 32
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

import dinov2_common as common

# Paths of images that could not be decoded during this run.
failures: list = []


def _records_signature(records) -> str:
    digest = hashlib.sha1()
    for path, label in records:
        digest.update(f"{path}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def _limit_per_class(records, per_class: int):
    counts = {}
    kept = []
    for path, label in records:
        n = counts.get(label, 0)
        if n >= per_class:
            continue
        counts[label] = n + 1
        kept.append((path, label))
    return kept


def _build_dataset(records):
    import torch
    from PIL import Image

    class _LeafDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, label = self.items[index]
            try:
                with Image.open(path) as img:
                    tensor = common.preprocess_pil(img)
                return tensor, label, 1
            except Exception as exc:  # unreadable / truncated image
                print(f"[warn] could not read {path}: {exc}")
                zeros = torch.zeros(3, common.DINOV2_CROP, common.DINOV2_CROP)
                return zeros, label, 0

    return _LeafDataset(records)


def extract_split(split: str, args) -> None:
    import torch
    from torch.utils.data import DataLoader

    records = common.split_records(split)
    if args.limit_per_class is not None:
        records = _limit_per_class(records, args.limit_per_class)

    if not records:
        raise SystemExit(f"No images found for split '{split}'.")

    signature = _records_signature(records)
    paths = common.feature_cache_paths(split, args.backbone, args.crop)
    feature_path = paths["features"]
    label_path = paths["labels"]
    path_list = paths["paths"]
    meta_path = paths["meta"]

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------
    processed = 0
    features = []
    labels = []
    kept_paths = []

    if meta_path.exists() and not args.overwrite:
        meta = common.load_json(meta_path)
        if (
            meta.get("complete")
            and meta.get("signature") == signature
            and meta.get("backbone") == args.backbone
            and meta.get("crop") == args.crop
            and feature_path.exists()
        ):
            print(f"[{split}] cached features already complete: {feature_path}")
            return
        if (
            meta.get("signature") == signature
            and meta.get("backbone") == args.backbone
            and meta.get("crop") == args.crop
            and feature_path.exists()
            and label_path.exists()
            and np.load(feature_path, mmap_mode="r").shape[1] == common.FEATURE_DIM
        ):
            processed = int(meta.get("records_processed", 0))
            features = list(np.load(feature_path, mmap_mode="r"))
            labels = list(np.load(label_path))
            kept_paths = common.load_json(path_list)["paths"]
            print(f"[{split}] resuming from records_processed={processed} "
                  f"({len(labels)} features kept)")

    if args.overwrite:
        processed = 0
        features, labels, kept_paths = [], [], []

    failures.clear()
    remaining = records[processed:]
    total = len(records)
    print(f"[{split}] {total} images, {len(remaining)} remaining, "
          f"batch_size={args.batch_size}, device={args.device}")

    if not remaining:
        _finalize(split, signature, features, labels, kept_paths, paths, args)
        return

    backbone = common.load_backbone(device=args.device)
    dataset = _build_dataset(remaining)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    kept_before_checkpoint = len(features)
    start = time.time()
    seen = 0

    for batch_images, batch_labels, batch_ok in loader:
        batch_features = common.extract_features_batch(backbone, batch_images, args.device)
        ok = batch_ok.numpy().astype(bool)
        for i in range(len(batch_labels)):
            processed += 1
            seen += 1
            if ok[i]:
                features.append(batch_features[i])
                labels.append(int(batch_labels[i]))
                kept_paths.append(remaining[seen - 1][0])
            else:
                # Record the file that failed so it can be inspected later.
                failures.append(remaining[seen - 1][0])

        elapsed = time.time() - start
        rate = seen / elapsed if elapsed > 0 else 0.0
        remaining_n = total - processed
        eta = remaining_n / rate if rate > 0 else float("inf")
        print(
            f"[{split}] {processed}/{total} "
            f"({rate:5.1f} img/s, ETA {_fmt(eta)})"
        )

        if seen >= args.save_every or processed == total:
            _checkpoint(split, signature, features, labels, kept_paths, paths,
                        processed, args)
            seen = 0
            start = time.time()

    _finalize(split, signature, features, labels, kept_paths, paths, args)


def _checkpoint(split, signature, features, labels, kept_paths, paths, processed, args):
    if not features:
        return
    np.save(paths["features"], np.asarray(features, dtype=np.float32))
    np.save(paths["labels"], np.asarray(labels, dtype=np.int64))
    common.save_json(paths["paths"], {"paths": kept_paths})
    common.save_json(paths["meta"], {
        "split": split,
        "backbone": args.backbone,
        "crop": args.crop,
        "feature_dim": common.FEATURE_DIM,
        "signature": signature,
        "records_processed": processed,
        "features_kept": len(features),
        "complete": False,
    })


def _finalize(split, signature, features, labels, kept_paths, paths, args):
    if not features:
        raise SystemExit(f"[{split}] no features extracted (all images unreadable?).")
    np.save(paths["features"], np.asarray(features, dtype=np.float32))
    np.save(paths["labels"], np.asarray(labels, dtype=np.int64))
    common.save_json(paths["paths"], {"paths": kept_paths})
    common.save_json(paths["meta"], {
        "split": split,
        "backbone": args.backbone,
        "crop": args.crop,
        "feature_dim": common.FEATURE_DIM,
        "signature": signature,
        "records_processed": len(kept_paths),
        "features_kept": len(features),
        "failures": failures,
        "complete": True,
    })
    print(f"[{split}] done: {len(features)} features -> {paths['features']}")


def _fmt(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        choices=["train", "validation", "test", "realworld"],
                        help="Which splits to extract. 'train' is the slow one.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Images per forward pass (CPU: 16-64 is reasonable).")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes. 0 is safest on Windows.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone", default=common.BACKBONE_NAME)
    parser.add_argument("--crop", type=int, default=common.DINOV2_CROP)
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Keep at most N images per class (smoke testing).")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Checkpoint interval, in images.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore any cached features and start over.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    for split in args.splits:
        extract_split(split, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
