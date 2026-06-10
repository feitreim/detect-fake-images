"""Download the real COCO images that pair with the fakes we have locally.

Generation is text->image, so the box only ever needs captions; the matching
real photos are pulled here, locally, straight from the COCO CDN. Keeps the
local dataset paired as the fake set grows. Resume-safe: skips reals already
on disk.

Usage:
    python -m src.sync_reals          # download reals for every local fake id
"""

import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


def download(url, out_path):
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fake-dir", default="data/fake_images")
    p.add_argument("--real-dir", default="data/real_images")
    p.add_argument("--urls", default="data/train_coco_urls.parquet")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    real_dir = Path(args.real_dir)
    real_dir.mkdir(parents=True, exist_ok=True)

    url_map = dict(zip(pd.read_parquet(args.urls)["image_id"].astype(int), pd.read_parquet(args.urls)["coco_url"]))

    fake_ids = {int(p.stem) for p in Path(args.fake_dir).glob("*.png")}
    todo = [i for i in fake_ids if not (real_dir / f"{i}.jpg").exists() and i in url_map]
    print(f"fakes={len(fake_ids)} reals_present={len(fake_ids) - len(todo)} to_download={len(todo)}")

    done, fails = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, url_map[i], real_dir / f"{i}.jpg"): i for i in todo}
        for f in as_completed(futures):
            try:
                f.result()
                done += 1
            except Exception:
                fails += 1
                if fails <= 3:
                    traceback.print_exc()

    print(f"downloaded {done} reals (fails={fails})")


if __name__ == "__main__":
    main()
