"""Generate fake videos via fal.ai image-to-video (Seedance 2.0).

Extracts the first frame of each real clip, uploads it, and generates
a continuation video. Resume-safe: skips clip_ids whose output mp4 exists.

Usage:
    export FAL_KEY=...
    python -m src.generate_fake --max-clips 10  # dry run
    python -m src.generate_fake                  # full run

Requires: pip install fal-client
"""
import argparse
import hashlib
import io
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import av
import numpy as np
import pandas as pd
import requests


def extract_first_frame(mp4_path):
    """Return the first frame as JPEG bytes."""
    from PIL import Image
    container = av.open(str(mp4_path))
    frame = next(container.decode(video=0))
    container.close()
    arr = frame.to_ndarray(format="rgb24")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def upload_image(jpeg_bytes):
    """Upload image to fal storage, return URL."""
    import fal_client
    url = fal_client.upload(jpeg_bytes, content_type="image/jpeg")
    return url


def generate_one(image_url, prompt, seed, duration, resolution):
    """Call Seedance 2.0 image-to-video, return video URL."""
    import fal_client
    result = fal_client.subscribe(
        "bytedance/seedance-2.0/image-to-video",
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "seed": seed,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": False,
        },
    )
    return result["video"]["url"], result.get("seed", seed)


def download_video(url, out_path):
    """Download video from URL to local path."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(resp.content)


def seed_for(clip_id, base):
    h = int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


def process_clip(row, out_dir, seed_base, duration, resolution):
    real_id = row["clip_id"]
    caption = str(row.get("caption", "")).strip() or "natural video motion"
    fake_id = real_id.replace("real/", "fake/")
    out_path = out_dir / (fake_id.split("/")[-1] + ".mp4")
    seed = seed_for(real_id, seed_base)

    if out_path.exists():
        return _row(fake_id, out_path, caption, seed, duration, resolution), None

    jpeg_bytes = extract_first_frame(row["mp4_path"])
    image_url = upload_image(jpeg_bytes)
    video_url, actual_seed = generate_one(image_url, caption, seed, duration, resolution)
    download_video(video_url, out_path)
    return _row(fake_id, out_path, caption, actual_seed, duration, resolution), None


def _row(clip_id, mp4_path, caption, seed, duration, resolution):
    return {
        "clip_id": clip_id,
        "mp4_path": str(Path(mp4_path).resolve()),
        "caption": caption,
        "model_name": "seedance-2.0",
        "model_version": "bytedance/seedance-2.0/image-to-video",
        "seed": int(seed),
        "duration": str(duration),
        "resolution": str(resolution),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-manifest", default="data/real_manifest.parquet")
    p.add_argument("--out-dir", default="data/fake")
    p.add_argument("--manifest", default="data/fake_manifest.parquet")
    p.add_argument("--duration", default="5", help="video duration in seconds, or 'auto'")
    p.add_argument("--resolution", default="480p", choices=["480p", "720p"])
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--workers", type=int, default=4, help="concurrent API requests")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--exclude-prefix", nargs="*", default=["celebv"],
                   help="skip clips whose filename starts with these prefixes")
    args = p.parse_args()

    real = pd.read_parquet(args.real_manifest)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load existing manifest to resume
    existing = set()
    rows = []
    if Path(args.manifest).exists():
        prev = pd.read_parquet(args.manifest)
        if "clip_id" in prev.columns and len(prev) > 0:
            existing = set(prev["clip_id"].tolist())
            rows = prev.to_dict("records")
            print(f"resuming: {len(existing)} already done")

    todo = [
        row for _, row in real.iterrows()
        if row["clip_id"].replace("real/", "fake/") not in existing
        and not any(row["clip_id"].split("/")[-1].startswith(p) for p in args.exclude_prefix)
    ]
    if args.max_clips:
        todo = todo[:args.max_clips]
    excluded = len(real) - len(todo) - len(existing)
    print(f"generating {len(todo)} clips ({excluded} excluded, {len(existing)} already done)")

    start = time.time()
    done = 0
    fails = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for row in todo:
            f = pool.submit(process_clip, row, out_dir, args.seed_base, args.duration, args.resolution)
            futures[f] = row["clip_id"]

        for f in as_completed(futures):
            clip_id = futures[f]
            try:
                result, _ = f.result()
                rows.append(result)
                done += 1
            except Exception as e:
                err_str = str(e)
                if "content_policy_violation" in err_str:
                    print(f"[policy skip] {clip_id}")
                    continue
                fails += 1
                if fails <= 10:
                    traceback.print_exc()
                print(f"[fail {fails}] {clip_id}: {e}")
                continue

            if done % 10 == 0:
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(todo) - done) / rate if rate > 0 else 0
                print(f"[{done}/{len(todo)}] {rate:.2f} clips/s  eta {remaining/60:.0f}m  fails={fails}")

            if done % args.save_every == 0:
                pd.DataFrame(rows).to_parquet(args.manifest, index=False)

    pd.DataFrame(rows).to_parquet(args.manifest, index=False)
    print(f"done: {len(rows)} clips → {args.manifest}  (fails: {fails})")


if __name__ == "__main__":
    main()
