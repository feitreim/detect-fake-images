"""Download real images + captions from COCO 2017 (via HuggingFace).

`phiyodr/coco2017` ships parquet metadata (captions embedded) but stores
images as `coco_url` pointers to the COCO CDN. We pull the parquet, then
download the original JPEGs — keeping their native single-pass JPEG
compression, which is what makes the codec-matched fake comparison honest.

Usage:
    python -m src.download_real_images                 # 5000 val images
    python -m src.download_real_images --limit 200     # quick subset
"""
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

REPO = "phiyodr/coco2017"
PARQUET = {
    "validation": "data/validation-00000-of-00001-e3c37e369512a3aa.parquet",
    "train": "data/train-00000-of-00001-0084e041f1902997.parquet",
}


def download_image(url, out_path):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def process_row(row, img_dir):
    image_id = int(row["image_id"])
    out_path = img_dir / f"{image_id}.jpg"
    caption = str(row["captions"][0]).strip() if len(row["captions"]) else ""
    if not out_path.exists():
        download_image(row["coco_url"], out_path)
    return {
        "image_id": image_id,
        "image_path": str(out_path.resolve()),
        "caption": caption,
        "width": int(row["width"]),
        "height": int(row["height"]),
        "source": "coco2017",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation", choices=["validation", "train"])
    p.add_argument("--out-dir", default="data/real_images")
    p.add_argument("--hf-cache", default="data/hf_cache")
    p.add_argument("--manifest", default="data/real_images_manifest.parquet")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    img_dir = Path(args.out_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(REPO, PARQUET[args.split], repo_type="dataset", cache_dir=args.hf_cache)
    meta = pd.read_parquet(parquet_path)
    if args.limit:
        meta = meta.iloc[: args.limit]
    print(f"{args.split}: {len(meta)} images")

    rows, fails = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, row, img_dir): int(row["image_id"]) for _, row in meta.iterrows()}
        for i, f in enumerate(as_completed(futures), 1):
            try:
                rows.append(f.result())
            except Exception:
                fails += 1
                if fails <= 5:
                    traceback.print_exc()
            if i % 500 == 0:
                print(f"  [{i}/{len(meta)}] fails={fails}")

    df = pd.DataFrame(rows).sort_values("image_id")
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.manifest, index=False)
    print(f"manifest: {len(df)} images → {args.manifest}  (fails: {fails})")


if __name__ == "__main__":
    main()
