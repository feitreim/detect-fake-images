"""Crop real + fake images into paired 64x64 patches, codec-matched.

Mirrors preprocess.py but for single images (C, H, W instead of T, C, H, W).
Real and fake share a per-image seed, so a given image_id always yields the
same crop coordinates — which means the only thing that differs between the
`jpeg` and `png` runs is the fake codec. That's what makes the ablation clean.

--fake-codec:
    jpeg  re-encode fake PNGs to JPEG q90 before cropping → matches the real
          photos' single-pass JPEG. The honest "generator artifacts" condition.
    png   leave fakes lossless → measures how much the detector leans on the
          JPEG-vs-clean compression gap. The ablation/upper-bound condition.

Reals are always decoded from their original COCO JPEG (never re-encoded, to
avoid double compression).

Usage:
    python -m src.preprocess_images --fake-codec jpeg --out-dir cache_images/jpeg
    python -m src.preprocess_images --fake-codec png  --out-dir cache_images/png
"""

import argparse
import hashlib
import io
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def image_rng(image_id):
    h = hashlib.md5(str(image_id).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def split_of(image_id, val_mod=10):
    h = int(hashlib.md5(str(image_id).encode()).hexdigest(), 16)
    return "val" if h % val_mod == 0 else "train"


def load_real(path):
    return np.array(Image.open(path).convert("RGB"))


def load_fake(path, codec, quality):
    img = Image.open(path).convert("RGB")
    if codec == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return np.array(img)


def resize_short_side(img_chw, target):
    _, H, W = img_chw.shape
    if min(H, W) <= target:
        return img_chw
    if H < W:
        new_h, new_w = target, round(W * target / H)
    else:
        new_h, new_w = round(H * target / W), target
    resized = F.interpolate(img_chw.float().unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).to(torch.uint8)


def sample_crops(img_hwc, n_crops, crop_size, resize_short, rng):
    """Resize short side, then draw n_crops random patches with the given rng.
    Same image_id → same rng → identical crops across the jpeg/png runs."""
    img = torch.from_numpy(img_hwc).permute(2, 0, 1)
    img = resize_short_side(img, max(resize_short, crop_size))
    _, H, W = img.shape
    if min(H, W) < crop_size:
        return []
    crops = []
    for _ in range(n_crops):
        t, l = rng.randint(0, H - crop_size), rng.randint(0, W - crop_size)
        crops.append(img[:, t : t + crop_size, l : l + crop_size].contiguous())
    return crops


def flush(buf, out_dir, split, idx):
    if not buf:
        return idx
    real = torch.stack([r["real"] for r in buf])
    fake = torch.stack([r["fake"] for r in buf])
    ids = [r["image_id"] for r in buf]
    (out_dir / split).mkdir(parents=True, exist_ok=True)
    torch.save({"real": real, "fake": fake, "image_ids": ids}, out_dir / split / f"shard-{idx:05d}.pt")
    return idx + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-manifest", default="data/real_images_manifest.parquet")
    p.add_argument("--fake-manifest", default="data/fake_images_manifest.parquet")
    p.add_argument(
        "--real-dir", default="data/real_images", help="locate reals by {dir}/{image_id}.jpg, ignoring manifest paths"
    )
    p.add_argument("--fake-dir", default="data/fake_images", help="locate fakes by {dir}/{image_id}.png")
    p.add_argument("--out-dir", default="cache_images/jpeg")
    p.add_argument("--fake-codec", default="jpeg", choices=["jpeg", "png"])
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--crops-per-image", type=int, default=8)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--resize-short", type=int, default=256)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    real_df = pd.read_parquet(args.real_manifest).set_index("image_id")
    fake_df = pd.read_parquet(args.fake_manifest).set_index("image_id")
    paired = real_df.join(fake_df, lsuffix="_real", rsuffix="_fake", how="inner")
    print(f"paired images: {len(paired)} (real={len(real_df)} fake={len(fake_df)})")

    image_ids = list(paired.index)
    random.Random(0).shuffle(image_ids)
    if args.limit:
        image_ids = image_ids[: args.limit]

    train_buf, val_buf = [], []
    train_idx, val_idx = 0, 0
    fails = 0

    real_dir, fake_dir = Path(args.real_dir), Path(args.fake_dir)
    for image_id in tqdm(image_ids):
        try:
            real_img = load_real(real_dir / f"{image_id}.jpg")
            fake_img = load_fake(fake_dir / f"{image_id}.png", args.fake_codec, args.jpeg_quality)

            rng = image_rng(image_id)
            real_crops = sample_crops(real_img, args.crops_per_image, args.crop_size, args.resize_short, rng)
            fake_crops = sample_crops(fake_img, args.crops_per_image, args.crop_size, args.resize_short, rng)
        except Exception:
            fails += 1
            if fails <= 5:
                traceback.print_exc()
            continue

        records = [{"real": r, "fake": f, "image_id": int(image_id)} for r, f in zip(real_crops, fake_crops)]
        split = split_of(image_id)
        if split == "train":
            train_buf.extend(records)
            if len(train_buf) >= args.shard_size:
                train_idx = flush(train_buf, out_dir, "train", train_idx)
                train_buf = []
        else:
            val_buf.extend(records)
            if len(val_buf) >= args.shard_size:
                val_idx = flush(val_buf, out_dir, "val", val_idx)
                val_buf = []

    flush(train_buf, out_dir, "train", train_idx)
    flush(val_buf, out_dir, "val", val_idx)
    print(f"done. {len(image_ids)} images, codec={args.fake_codec}, failures: {fails}")


if __name__ == "__main__":
    main()
