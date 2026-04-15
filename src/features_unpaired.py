"""Extract features directly from mp4 files for the unpaired experiment.

Train real = real clips with no fake counterpart
Train fake = 80% of fake clips
Val real = held-out real clips (from unpaired set)
Val fake = 20% of fake clips

No content overlap between fake source clips and real training data.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import extract_features
from src.preprocess import decode_video, sample_triplets, resize_short_side, clip_rng

import torch


def features_from_video(mp4_path, clip_id, n_triplets=10, n_crops=4,
                        crop_size=64, resize_short=80, k_choices=(1, 2, 4)):
    frames_thwc = decode_video(mp4_path)
    if frames_thwc.shape[0] < 8:
        return []
    rng = clip_rng(clip_id)
    triplets = sample_triplets(frames_thwc.shape[0], n_triplets, k_choices, rng)
    if not triplets:
        return []

    feats = []
    for t0, t1, t2, k in triplets:
        tri = torch.from_numpy(frames_thwc[[t0, t1, t2]]).permute(0, 3, 1, 2)
        short = min(tri.shape[-2], tri.shape[-1])
        if short < crop_size:
            continue
        tri = resize_short_side(tri, max(resize_short, crop_size))
        _, _, H, W = tri.shape
        for _ in range(n_crops):
            top = rng.randint(0, H - crop_size)
            left = rng.randint(0, W - crop_size)
            cropped = tri[:, :, top:top + crop_size, left:left + crop_size].contiguous()
            feats.append(extract_features(cropped))
    return feats


def val_split(clip_id, val_frac=0.2):
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16)
    return h % 100 < int(val_frac * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-manifest", default="data/real_manifest.parquet")
    p.add_argument("--fake-manifest", default="data/fake_manifest.parquet")
    p.add_argument("--out", default="cache/features_unpaired.npz")
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--triplets", type=int, default=10)
    p.add_argument("--crops", type=int, default=4)
    args = p.parse_args()

    real_df = pd.read_parquet(args.real_manifest)
    fake_df = pd.read_parquet(args.fake_manifest)

    # find which real clips were used to generate fakes
    fake_base_ids = set(fake_df["clip_id"].str.replace("fake/", "", n=1))
    real_df["base_id"] = real_df["clip_id"].str.replace("real/", "", n=1)

    unpaired_real = real_df[~real_df["base_id"].isin(fake_base_ids)]
    print(f"unpaired real clips: {len(unpaired_real)} (excluded {len(real_df) - len(unpaired_real)} paired)")

    # split unpaired reals into train/val
    train_real_rows = [r for _, r in unpaired_real.iterrows() if not val_split(r["clip_id"])]
    val_real_rows = [r for _, r in unpaired_real.iterrows() if val_split(r["clip_id"])]

    # split fakes into train/val
    train_fake_rows = [r for _, r in fake_df.iterrows() if not val_split(r["clip_id"])]
    val_fake_rows = [r for _, r in fake_df.iterrows() if val_split(r["clip_id"])]

    print(f"train: {len(train_real_rows)} real + {len(train_fake_rows)} fake")
    print(f"val:   {len(val_real_rows)} real + {len(val_fake_rows)} fake")

    def extract_all(rows, desc):
        all_feats = []
        fails = 0
        for row in tqdm(rows, desc=desc):
            try:
                feats = features_from_video(
                    row["mp4_path"], row["clip_id"],
                    n_triplets=args.triplets, n_crops=args.crops,
                    crop_size=args.crop_size,
                )
                all_feats.extend(feats)
            except Exception:
                fails += 1
        if fails:
            print(f"  {fails} failures in {desc}")
        return np.stack(all_feats) if all_feats else np.empty((0, 25))

    print("extracting features...")
    X_train_real = extract_all(train_real_rows, "train real")
    X_train_fake = extract_all(train_fake_rows, "train fake")
    X_val_real = extract_all(val_real_rows, "val real")
    X_val_fake = extract_all(val_fake_rows, "val fake")

    X_train = np.concatenate([X_train_real, X_train_fake])
    y_train = np.concatenate([np.zeros(len(X_train_real)), np.ones(len(X_train_fake))])
    X_val = np.concatenate([X_val_real, X_val_fake])
    y_val = np.concatenate([np.zeros(len(X_val_real)), np.ones(len(X_val_fake))])

    np.savez(args.out, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print(f"saved {args.out}: train={len(X_train)} ({y_train.sum():.0f} fake) val={len(X_val)} ({y_val.sum():.0f} fake)")


if __name__ == "__main__":
    main()
