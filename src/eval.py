"""Evaluate saved train.py checkpoints: per-crop and per-image metrics on the val split.

Each image's call is the average sigmoid score over its first --crops cached
crops (default: all of them, 8 per image in the full cache).

Usage:
    python -m src.eval runs_images/cnn/final.pt runs_images/vit/final.pt
    python -m src.eval runs_images/cnn/final.pt --crops 4
"""

import argparse
from pathlib import Path

import torch

from src.train import ImageCNN, ImageViT, evaluate, pick_device, print_confusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+", help="final.pt files saved by train.py")
    p.add_argument("--cache-dir", default=None, help="defaults to the cache dir the model was trained on")
    p.add_argument("--crops", type=int, default=None, help="crops averaged per image (default: all cached)")
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    device = pick_device()
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        train_args = ckpt["args"]
        model = (ImageCNN() if train_args["model"] == "cnn" else ImageViT()).to(device)
        model.load_state_dict(ckpt["model"])
        val_dir = Path(args.cache_dir or train_args["cache_dir"]) / "val"
        m = evaluate(model, val_dir, device, args.crops, args.batch_size)
        c, im = m["crop"], m["image"]
        print(f"\n[{path}] model={train_args['model']}")
        print(f"  crop:  acc={c['acc']:.3f}  bal={c['bal']:.3f}  auc={c['auc']:.3f}  (n={c['n']})")
        print(
            f"  image: acc={im['acc']:.3f}  bal={im['bal']:.3f}  auc={im['auc']:.3f}  "
            f"(n={im['n']}, avg over {args.crops or 'all'} crops)"
        )
        print_confusion(im["cm"])


if __name__ == "__main__":
    main()
