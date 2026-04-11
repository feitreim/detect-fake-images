import argparse
import hashlib
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.io import read_video
from torchvision.transforms.v2.functional import resize
from tqdm import tqdm


def clip_rng(clip_id):
    h = hashlib.md5(clip_id.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def split_of(clip_id, val_mod=20):
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16)
    return "val" if h % val_mod == 0 else "train"


def sample_triplets(num_frames, n_triplets, k_choices, rng):
    out = []
    for _ in range(n_triplets):
        k = rng.choice(k_choices)
        span = 2 * k
        if num_frames <= span:
            continue
        t0 = rng.randint(0, num_frames - span - 1)
        out.append((t0, t0 + k, t0 + 2 * k, k))
    return out


def process_clip(mp4_path, label, clip_id, n_triplets, n_crops, k_choices, crop_size, resize_short):
    frames_thwc, _, _ = read_video(str(mp4_path), pts_unit="sec", output_format="THWC")
    if frames_thwc.shape[0] < 8:
        return []
    rng = clip_rng(clip_id)
    triplets = sample_triplets(frames_thwc.shape[0], n_triplets, k_choices, rng)
    if not triplets:
        return []

    records = []
    for t0, t1, t2, k in triplets:
        tri = frames_thwc[[t0, t1, t2]].permute(0, 3, 1, 2)  # (3, C, H, W)
        short = min(tri.shape[-2], tri.shape[-1])
        if short < crop_size:
            continue
        target = max(resize_short, crop_size)
        resized = resize(tri, size=target, antialias=True)
        _, _, H, W = resized.shape
        if H < crop_size or W < crop_size:
            continue
        for _ in range(n_crops):
            top = rng.randint(0, H - crop_size)
            left = rng.randint(0, W - crop_size)
            cropped = resized[:, :, top:top + crop_size, left:left + crop_size].contiguous()
            records.append({
                "frames": cropped.to(torch.uint8) if cropped.dtype != torch.uint8 else cropped,
                "label": label,
                "clip_id": clip_id,
                "temporal_k": k,
            })
    return records


def flush_shard(buf, out_dir, split, idx):
    if not buf:
        return idx
    frames = torch.stack([r["frames"] for r in buf])
    labels = torch.tensor([r["label"] for r in buf], dtype=torch.int8)
    temporal_k = torch.tensor([r["temporal_k"] for r in buf], dtype=torch.int8)
    clip_ids = [r["clip_id"] for r in buf]
    path = out_dir / split / f"shard-{idx:05d}.pt"
    torch.save(
        {"frames": frames, "labels": labels, "temporal_k": temporal_k, "clip_ids": clip_ids},
        path,
    )
    return idx + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-manifest", default="data/real_manifest.parquet")
    p.add_argument("--fake-manifest", default="data/fake_manifest.parquet")
    p.add_argument("--out-dir", default="cache")
    p.add_argument("--real-triplets-per-clip", type=int, default=6)
    p.add_argument("--fake-triplets-per-clip", type=int, default=3)
    p.add_argument("--crops-per-triplet", type=int, default=8)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--resize-short", type=int, default=80,
                   help="resize shorter side to this before random cropping")
    p.add_argument("--shard-size", type=int, default=10000)
    p.add_argument("--k-choices", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    all_clips = []
    if Path(args.real_manifest).exists():
        real = pd.read_parquet(args.real_manifest)
        for _, r in real.iterrows():
            all_clips.append((r["mp4_path"], 0, r["clip_id"], args.real_triplets_per_clip))
    if Path(args.fake_manifest).exists():
        fake = pd.read_parquet(args.fake_manifest)
        for _, r in fake.iterrows():
            all_clips.append((r["mp4_path"], 1, r["clip_id"], args.fake_triplets_per_clip))

    if not all_clips:
        raise SystemExit("no clips found — run download_real.py / generate_fake.py first")

    random.Random(0).shuffle(all_clips)
    if args.limit:
        all_clips = all_clips[:args.limit]

    train_buf, val_buf = [], []
    train_idx, val_idx = 0, 0
    fails = 0

    for mp4_path, label, clip_id, n_triplets in tqdm(all_clips):
        try:
            records = process_clip(
                mp4_path, label, clip_id, n_triplets,
                args.crops_per_triplet, args.k_choices,
                args.crop_size, args.resize_short,
            )
        except Exception:
            fails += 1
            if fails <= 5:
                traceback.print_exc()
            continue
        split = split_of(clip_id)
        buf = val_buf if split == "val" else train_buf
        buf.extend(records)
        if split == "train" and len(train_buf) >= args.shard_size:
            train_idx = flush_shard(train_buf, out_dir, "train", train_idx)
            train_buf = []
        elif split == "val" and len(val_buf) >= args.shard_size:
            val_idx = flush_shard(val_buf, out_dir, "val", val_idx)
            val_buf = []

    flush_shard(train_buf, out_dir, "train", train_idx)
    flush_shard(val_buf, out_dir, "val", val_idx)
    print(f"done. failures: {fails}")


if __name__ == "__main__":
    main()
