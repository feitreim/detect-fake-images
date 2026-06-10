"""Extract hand-crafted features from image crops for logistic regression.

The image analogue of features.py: same statistics, minus the 4 temporal
frame-diff features that only exist for video. 21 features per (C=3, H=64, W=64) crop:
- Per-channel pixel mean/std (6)
- DCT high-freq energy per channel (3)
- Patch-level variance stats: mean/std of 8x8 patch variances (6)
- Noise residual magnitude: blur, subtract, measure mean abs per channel (3)
- Edge energy: Sobel magnitude mean per channel (3)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter, sobel
from tqdm import tqdm

FEATURE_NAMES = [
    "R_mean",
    "R_std",
    "G_mean",
    "G_std",
    "B_mean",
    "B_std",
    "dct_hi_R",
    "dct_hi_G",
    "dct_hi_B",
    "pvar_mean_R",
    "pvar_std_R",
    "pvar_mean_G",
    "pvar_std_G",
    "pvar_mean_B",
    "pvar_std_B",
    "noise_R",
    "noise_G",
    "noise_B",
    "edge_R",
    "edge_G",
    "edge_B",
]


def extract_features(crop):
    """crop: uint8 (C=3, H, W) -> feature vector (float32)."""
    x = crop.numpy().astype(np.float32) / 255.0  # (C, H, W)
    C, H, W = x.shape
    feats = []

    # 1. per-channel pixel mean/std (6)
    for c in range(C):
        feats.append(x[c].mean())
        feats.append(x[c].std())

    # 2. DCT high-freq energy per channel (3)
    for c in range(C):
        dct = dctn(x[c], norm="ortho")
        total = (dct**2).sum()
        hi = (dct[H // 2 :, W // 2 :] ** 2).sum()
        feats.append(hi / (total + 1e-10))

    # 3. patch variance stats — 8x8 patches (6)
    ps = 8
    for c in range(C):
        frame = x[c]
        patches = frame[: H - H % ps, : W - W % ps].reshape(H // ps, ps, W // ps, ps)
        patch_vars = patches.var(axis=(1, 3)).flatten()
        feats.append(patch_vars.mean())
        feats.append(patch_vars.std())

    # 4. noise residual — gaussian blur then subtract (3)
    for c in range(C):
        noise = x[c] - gaussian_filter(x[c], sigma=1.5)
        feats.append(np.abs(noise).mean())

    # 5. edge energy — sobel (3)
    for c in range(C):
        sx, sy = sobel(x[c], axis=0), sobel(x[c], axis=1)
        feats.append(np.sqrt(sx**2 + sy**2).mean())

    return np.array(feats, dtype=np.float32)


def process_shards(shard_dir):
    shards = sorted(Path(shard_dir).glob("shard-*.pt"))
    X_real, X_fake = [], []
    for path in tqdm(shards, desc=str(shard_dir)):
        data = torch.load(path, map_location="cpu", weights_only=True)
        for crop in data["real"]:
            X_real.append(extract_features(crop))
        for crop in data["fake"]:
            X_fake.append(extract_features(crop))
    return np.stack(X_real), np.stack(X_fake)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache_images/jpeg")
    p.add_argument("--out", default="cache_images/features_jpeg.npz")
    args = p.parse_args()

    print("extracting train features...")
    train_real, train_fake = process_shards(Path(args.cache_dir) / "train")
    print("extracting val features...")
    val_real, val_fake = process_shards(Path(args.cache_dir) / "val")

    X_train = np.concatenate([train_real, train_fake])
    y_train = np.concatenate([np.zeros(len(train_real)), np.ones(len(train_fake))])
    X_val = np.concatenate([val_real, val_fake])
    y_val = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, feature_names=np.array(FEATURE_NAMES)
    )
    print(f"saved {args.out}: train={len(X_train)} val={len(X_val)} features={X_train.shape[1]}")


if __name__ == "__main__":
    main()
