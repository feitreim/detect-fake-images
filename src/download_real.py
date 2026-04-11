"""Download OpenVid-1M shards from HuggingFace and build a manifest.

OpenVid-1M ships as:
  - data/train/OpenVid-1M.csv  (metadata: video, caption, aesthetic_score, motion_score, ...)
  - OpenVid_part{0..N}.zip     (sharded video archives)

This script downloads one or more parts, extracts mp4s to --out-dir/videos,
and writes real_manifest.parquet joining filenames → captions.
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO = "nkp37/OpenVid-1M"
METADATA_PATH = "data/train/OpenVid-1M.csv"


def download_metadata(cache_dir):
    print("downloading metadata CSV...")
    path = hf_hub_download(
        repo_id=REPO,
        filename=METADATA_PATH,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(path)


def download_part(part_idx, cache_dir):
    name = f"OpenVid_part{part_idx}.zip"
    print(f"downloading {name}...")
    path = hf_hub_download(
        repo_id=REPO,
        filename=name,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(path)


def extract_zip(zip_path, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".mp4")]
        for m in tqdm(members, desc=f"extract {zip_path.name}"):
            # flatten: drop any internal directory structure
            out_path = target_dir / Path(m).name
            if out_path.exists():
                continue
            with zf.open(m) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    return [target_dir / Path(m).name for m in members]


def build_manifest(video_dir, metadata_csv, out_parquet):
    print(f"reading metadata {metadata_csv}...")
    meta = pd.read_csv(metadata_csv)
    # metadata 'video' column holds filenames like "---_iRTHryQ_13_0to241.mp4"
    meta = meta.set_index("video")

    rows = []
    for mp4 in sorted(video_dir.glob("*.mp4")):
        fname = mp4.name
        if fname not in meta.index:
            continue
        row = meta.loc[fname]
        # a filename may have multiple caption rows; take first
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append({
            "clip_id": f"real/{fname[:-4]}",
            "mp4_path": str(mp4.resolve()),
            "caption": str(row.get("caption", "")),
            "source": "openvid",
            "original_fps": float(row.get("fps", 0.0) or 0.0),
            "duration_s": float(row.get("seconds", 0.0) or 0.0),
        })

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"manifest: {len(df)} clips → {out_parquet}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/real")
    p.add_argument("--hf-cache", default="data/hf_cache")
    p.add_argument("--parts", type=int, nargs="+", default=[0],
                   help="which OpenVid_partN.zip files to download (default: just part 0)")
    p.add_argument("--manifest", default="data/real_manifest.parquet")
    p.add_argument("--keep-zips", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = Path(args.hf_cache)
    hf_cache.mkdir(parents=True, exist_ok=True)

    metadata_csv = download_metadata(hf_cache)

    for part in args.parts:
        try:
            zip_path = download_part(part, hf_cache)
        except Exception as e:
            print(f"part {part} download failed: {e}")
            continue
        extract_zip(zip_path, video_dir)
        if not args.keep_zips:
            try:
                zip_path.unlink()
                print(f"deleted {zip_path.name}")
            except OSError:
                pass

    build_manifest(video_dir, metadata_csv, Path(args.manifest))


if __name__ == "__main__":
    main()
