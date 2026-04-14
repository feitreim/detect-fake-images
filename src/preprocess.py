import argparse
import hashlib
import random
import traceback
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def clip_rng(clip_id):
    h = hashlib.md5(clip_id.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def split_of(clip_id, val_mod=20):
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16)
    return "val" if h % val_mod == 0 else "train"


def decode_video(mp4_path):
    container = av.open(str(mp4_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames) if frames else np.empty((0, 0, 0, 3), dtype=np.uint8)


def resize_short_side(frames_tchw, target):
    _, _, H, W = frames_tchw.shape
    if min(H, W) <= target:
        return frames_tchw
    if H < W:
        new_h, new_w = target, int(round(W * target / H))
    else:
        new_h, new_w = int(round(H * target / W)), target
    return F.interpolate(frames_tchw.float(), size=(new_h, new_w), mode="bilinear", align_corners=False).to(torch.uint8)


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


def extract_crops(frames_thwc, triplets, n_crops, crop_size, resize_short, rng):
    """Given decoded frames and triplet indices, return list of (cropped_tensor, k) pairs."""
    crops = []
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
            crops.append((cropped, k))
    return crops


def process_pair(real_path, fake_path, base_id, n_triplets, n_crops, k_choices, crop_size, resize_short):
    """Process a real/fake pair. Returns list of {real_frames, fake_frames, ...} dicts."""
    real_frames = decode_video(real_path)
    fake_frames = decode_video(fake_path)
    if real_frames.shape[0] < 8 or fake_frames.shape[0] < 8:
        return []

    # sample triplets valid for the shorter video
    min_len = min(real_frames.shape[0], fake_frames.shape[0])
    rng = clip_rng(base_id)
    triplets = sample_triplets(min_len, n_triplets, k_choices, rng)
    if not triplets:
        return []

    real_crops = extract_crops(real_frames, triplets, n_crops, crop_size, resize_short, rng)
    # reset rng to get same crop positions for fake
    rng = clip_rng(base_id)
    sample_triplets(min_len, n_triplets, k_choices, rng)  # advance rng past triplet sampling
    fake_crops = extract_crops(fake_frames, triplets, n_crops, crop_size, resize_short, rng)

    records = []
    for (real_crop, k), (fake_crop, _) in zip(real_crops, fake_crops):
        records.append({
            "real_frames": real_crop,
            "fake_frames": fake_crop,
            "clip_id": base_id,
            "temporal_k": k,
        })
    return records


def flush_paired_shard(buf, out_dir, split, idx):
    if not buf:
        return idx
    real = torch.stack([r["real_frames"] for r in buf])
    fake = torch.stack([r["fake_frames"] for r in buf])
    temporal_k = torch.tensor([r["temporal_k"] for r in buf], dtype=torch.int8)
    clip_ids = [r["clip_id"] for r in buf]
    path = out_dir / split / f"shard-{idx:05d}.pt"
    torch.save({"real_frames": real, "fake_frames": fake, "temporal_k": temporal_k, "clip_ids": clip_ids}, path)
    return idx + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-manifest", default="data/real_manifest.parquet")
    p.add_argument("--fake-manifest", default="data/fake_manifest.parquet")
    p.add_argument("--out-dir", default="cache")
    p.add_argument("--triplets-per-clip", type=int, default=10)
    p.add_argument("--crops-per-triplet", type=int, default=8)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--resize-short", type=int, default=80)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--k-choices", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)

    real_df = pd.read_parquet(args.real_manifest)
    fake_df = pd.read_parquet(args.fake_manifest)

    # join on base clip name (strip real/ and fake/ prefixes)
    real_df["base_id"] = real_df["clip_id"].str.replace("real/", "", n=1)
    fake_df["base_id"] = fake_df["clip_id"].str.replace("fake/", "", n=1)
    paired = real_df.set_index("base_id").join(fake_df.set_index("base_id"), lsuffix="_real", rsuffix="_fake", how="inner")
    print(f"paired clips: {len(paired)} (real={len(real_df)} fake={len(fake_df)})")

    if len(paired) == 0:
        raise SystemExit("no paired clips found")

    pairs = list(paired.iterrows())
    random.Random(0).shuffle(pairs)
    if args.limit:
        pairs = pairs[:args.limit]

    train_buf, val_buf = [], []
    train_idx, val_idx = 0, 0
    fails = 0

    for base_id, row in tqdm(pairs):
        try:
            records = process_pair(
                row["mp4_path_real"], row["mp4_path_fake"], base_id,
                args.triplets_per_clip, args.crops_per_triplet, args.k_choices,
                args.crop_size, args.resize_short,
            )
        except Exception:
            fails += 1
            if fails <= 5:
                traceback.print_exc()
            continue

        split = split_of(base_id)
        buf = val_buf if split == "val" else train_buf
        buf.extend(records)
        if split == "train" and len(train_buf) >= args.shard_size:
            train_idx = flush_paired_shard(train_buf, out_dir, "train", train_idx)
            train_buf = []
        elif split == "val" and len(val_buf) >= args.shard_size:
            val_idx = flush_paired_shard(val_buf, out_dir, "val", val_idx)
            val_buf = []

    flush_paired_shard(train_buf, out_dir, "train", train_idx)
    flush_paired_shard(val_buf, out_dir, "val", val_idx)
    print(f"done. {len(paired)} pairs, failures: {fails}")


if __name__ == "__main__":
    main()
