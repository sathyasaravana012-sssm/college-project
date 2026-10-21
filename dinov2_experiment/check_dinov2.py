"""Verify the DINOv2 experiment environment and pretrained model loading.

Run this first:

    .venv\\Scripts\\python.exe check_dinov2.py

It prints versions, confirms no GPU is required, loads the frozen
dinov2_vits14 backbone (downloading it on first run), and runs one dummy
forward pass to prove feature extraction works. It writes nothing to the
PlantVillage dataset and touches no existing project file.
"""

from __future__ import annotations

import os
import platform
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main() -> int:
    import torch
    import numpy as np

    print("=" * 70)
    print("DINOv2 environment check")
    print("=" * 70)
    print(f"Python           : {platform.python_version()} ({sys.executable})")
    print(f"Platform         : {platform.platform()}")
    print(f"torch            : {torch.__version__}")
    print(f"numpy            : {np.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()} (CPU-only is fine)")

    try:
        import sklearn
        print(f"scikit-learn     : {sklearn.__version__}")
    except Exception as exc:  # pragma: no cover - informative only
        print(f"scikit-learn     : NOT INSTALLED ({exc})")

    print(f"Threads          : {torch.get_num_threads()}")

    print("\nLoading frozen DINOv2 backbone (dinov2_vits14)...")
    print("First run downloads code from GitHub + weights from dl.fbaipublicfiles.com.")
    print("This needs internet access once; afterwards it is cached and offline-safe.")

    import dinov2_common as common

    backbone = common.load_backbone(device="cpu")
    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone loaded. Parameters: {num_params:,}")

    rng = np.random.default_rng(0)
    dummy = torch.from_numpy(rng.standard_normal((2, 3, common.DINOV2_CROP, common.DINOV2_CROP)).astype("float32"))

    with torch.no_grad():
        features = common.extract_features_batch(backbone, dummy, device="cpu")

    print(f"Feature shape    : {features.shape} (expected (2, {common.FEATURE_DIM}))")

    ok = features.shape == (2, common.FEATURE_DIM) and np.isfinite(features).all()
    print("\nRESULT:", "OK - backbone works." if ok else "FAILED - unexpected feature shape/values.")

    print("\nClass mapping sanity:")
    names = common.class_names()
    print(f"  found {len(names)} classes; first={names[0]!r} last={names[-1]!r}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
