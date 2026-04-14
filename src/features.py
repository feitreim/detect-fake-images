"""Extract hand-crafted features from paired shards for logistic regression.

Features per crop (T=3, C=3, H=64, W=64):
- Per-channel pixel mean/std (6)
- Per-channel frame-to-frame diff mean/std (4) — temporal flicker
- DCT energy in high-freq bands per channel (3)
- Patch-level variance stats: mean/std of 8x8 patch variances (6)
- Noise residual magnitude: blur original, subtract, measure std per channel (3)
- Edge energy: Sobel magnitude mean per channel (3)
Total: 25 features per crop
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn
from tqdm import tqdm


def extract_features(frames):
    """frames: uint8 (T=3, C=3, H, W) -> feature vector (float32)."""
    x = frames.numpy().astype(np.float32) / 255.0  # (3, 3, H, W)
    T, C, H, W = x.shape
    feats = []

    # 1. per-channel pixel mean/std (6)
    for c in range(C):
        feats.append(x[:, c].mean())
        feats.append(x[:, c].std())

    # 2. temporal frame diff mean/std (4)
    diffs = np.abs(x[1:] - x[:-1])  # (2, C, H, W)
    feats.append(diffs.mean())
    feats.append(diffs.std())
    feats.append(diffs.mean(axis=(2, 3)).mean())  # mean per-pixel diff
    feats.append(diffs.mean(axis=(2, 3)).std())

    # 3. DCT high-freq energy per channel (3)
    for c in range(C):
        avg_frame = x[:, c].mean(axis=0)  # (H, W)
        dct = dctn(avg_frame, norm="ortho")
        total = (dct ** 2).sum()
        # high freq = bottom-right quadrant
        hi = (dct[H // 2:, W // 2:] ** 2).sum()
        feats.append(hi / (total + 1e-10))

    # 4. patch variance stats — 8x8 patches (6)
    ps = 8
    for c in range(C):
        avg_frame = x[:, c].mean(axis=0)
        patches = avg_frame[:H - H % ps, :W - W % ps].reshape(H // ps, ps, W // ps, ps)
        patch_vars = patches.var(axis=(1, 3)).flatten()
        feats.append(patch_vars.mean())
        feats.append(patch_vars.std())

    # 5. noise residual — gaussian blur then subtract (3)
    from scipy.ndimage import gaussian_filter
    for c in range(C):
        avg_frame = x[:, c].mean(axis=0)
        blurred = gaussian_filter(avg_frame, sigma=1.5)
        noise = avg_frame - blurred
        feats.append(np.abs(noise).mean())

    # 6. edge energy — sobel (3)
    from scipy.ndimage import sobel
    for c in range(C):
        avg_frame = x[:, c].mean(axis=0)
        sx = sobel(avg_frame, axis=0)
        sy = sobel(avg_frame, axis=1)
        feats.append(np.sqrt(sx ** 2 + sy ** 2).mean())

    return np.array(feats, dtype=np.float32)


def process_shards(shard_dir):
    """Load paired shards, extract features for both real and fake."""
    shards = sorted(Path(shard_dir).glob("shard-*.pt"))
    X_real, X_fake = [], []
    for path in tqdm(shards, desc=str(shard_dir)):
        data = torch.load(path, map_location="cpu", weights_only=True)
        real_frames = data["real_frames"]
        fake_frames = data["fake_frames"]
        for i in range(len(real_frames)):
            X_real.append(extract_features(real_frames[i]))
            X_fake.append(extract_features(fake_frames[i]))
    return np.stack(X_real), np.stack(X_fake)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="cache/features.npz")
    args = p.parse_args()

    print("extracting train features...")
    train_real, train_fake = process_shards(Path(args.cache_dir) / "train")
    print("extracting val features...")
    val_real, val_fake = process_shards(Path(args.cache_dir) / "val")

    # stack into X, y
    X_train = np.concatenate([train_real, train_fake])
    y_train = np.concatenate([np.zeros(len(train_real)), np.ones(len(train_fake))])
    X_val = np.concatenate([val_real, val_fake])
    y_val = np.concatenate([np.zeros(len(val_real)), np.ones(len(val_fake))])

    np.savez(args.out, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print(f"saved {args.out}: train={len(X_train)} val={len(X_val)} features={X_train.shape[1]}")


if __name__ == "__main__":
    main()
