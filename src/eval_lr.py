"""Image-level (crop-averaged) eval for the logistic-regression baseline.

train_lr.py only reports per-crop metrics; the CNN/ViT get a "free test-time
ensemble" by averaging the 8 crop scores per image (train.py:evaluate). This
applies that same averaging to the LR so the comparison is apples-to-apples.

The features npz drops image_id, so we re-extract features straight from the
val crop shards (which keep image_ids) and group by (image_id, label).

Usage:
    python -m src.eval_lr --cache-dir cache_images/jpeg_full \
        --features cache_images/features_jpeg_full.npz
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features_images import extract_features
from src.train import metrics, print_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache_images/jpeg_full")
    p.add_argument("--features", default="cache_images/features_jpeg_full.npz")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--crops", type=int, default=None, help="crops averaged per image (default: all)")
    args = p.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    bad = np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1)
    X_train, y_train = X_train[~bad], y_train[~bad]

    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=args.C, max_iter=1000, solver="lbfgs", class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_train)

    crop_labels, crop_probs, per_image = [], [], defaultdict(list)
    for shard in sorted((Path(args.cache_dir) / "val").glob("shard-*.pt")):
        d = torch.load(shard, map_location="cpu", weights_only=True)
        for label, key in ((0, "real"), (1, "fake")):
            feats = np.stack([extract_features(c) for c in d[key]])
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            probs = clf.predict_proba(scaler.transform(feats))[:, 1]
            crop_labels.extend([label] * len(probs))
            crop_probs.extend(probs.tolist())
            for image_id, prob in zip(d["image_ids"], probs.tolist()):
                per_image[(image_id, label)].append(prob)

    image_labels = np.array([label for _, label in per_image])
    image_probs = np.array([np.mean(p[: args.crops]) for p in per_image.values()])

    c = metrics(np.array(crop_labels), np.array(crop_probs))
    im = metrics(image_labels, image_probs)
    print(f"[logistic regression, C={args.C}]")
    print(f"  crop:  acc={c['acc']:.3f}  bal={c['bal']:.3f}  auc={c['auc']:.3f}  (n={c['n']})")
    print(f"  image: acc={im['acc']:.3f}  bal={im['bal']:.3f}  auc={im['auc']:.3f}  "
          f"(n={im['n']}, avg over {args.crops or 'all'} crops)")
    print_confusion(im["cm"])


if __name__ == "__main__":
    main()
