# Real Video Data

## Goal

End up with ~40k short real-video clips plus their text captions, stored locally in a uniform format.

## Primary source: OpenVid-1M

HuggingFace: `nkp37/OpenVid-1M`. CC-BY-4.0. Download strategy:

1. Pull only the metadata CSV first — per-clip URL, start/end seconds, caption.
2. Filter to the HD subset and shuffle-sample ~40k rows.
3. Use `video2dataset` (pip) to download those URLs in parallel, one worker per core, producing `.mp4` files in WebDataset shards (`*.tar`).
4. Oversample by ~20% to absorb 404s.

## Backstop source: MSR-VTT

HuggingFace: `friedrichor/MSR-VTT`. ~10k pre-chunked clips with captions. Used if OpenVid downloads are flaky or a backup is needed for the sanity split.

## Output schema

Single `real_manifest.parquet`:

```
clip_id         str
mp4_path        str
caption         str
source          str   # "openvid" | "msrvtt"
original_fps    float
duration_s      float
```

## Script

`src/download_real.py`

Args:
- `--target-count` (default 40000)
- `--out-dir` (default `data/real/`)
- `--shard-size` (default 1000)
- `--source` (`openvid` | `msrvtt` | `both`)

Resume-safe: skips shards that already exist.

## Disk budget

~400 GB raw. After `preprocess.py` extracts tensor shards we can delete the raw MP4s → ~150 GB steady state.

## Known risks

- Some OpenVid source URLs will 404. Oversample handles this.
- No widespread AI-generation contamination in OpenVid at our scale.
- MSR-VTT test split annotations are restricted; we only use the train split.
