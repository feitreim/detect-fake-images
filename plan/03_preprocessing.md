# Preprocessing

## Goal

Convert every MP4 (real + fake) into pre-cropped, pre-normalized uint8 tensors on disk so training I/O is trivial.

## Why up-front

Model inputs are tiny (3×3×256×256 uint8 = ~590 KB). Raw video decode + random crop is the expensive part. Doing it once turns training into a linear scan of tensor shards.

## Per-clip pipeline

1. Decode with `decord` (preferred for speed) or `torchvision.io.read_video` (fallback if `decord` is flaky on macOS).
2. Resize shorter side to 288, preserving aspect ratio (leaves margin for random 256 crop).
3. Sample `N` random `(t, t+k, t+2k)` triplets per clip. `k ∈ {1, 2, 4}` drawn uniformly per triplet.
   - Real clips: `N = 6` (plenty of frames to sample from).
   - Fake clips: `N = 3` (only 30 frames available).
4. For each triplet, take `M = 2` random 256×256 crops.
5. Store as uint8 in shards.

## Shard format

WebDataset tar shards, ~10k triplets/shard. Each record:

```
{
  frames:         uint8[3, 3, 256, 256]
  label:          int8    # 0 = real, 1 = fake
  source_clip_id: str
  temporal_k:     int8
}
```

Output locations:
- `cache/train/shard-{0000..}.tar`
- `cache/val/shard-{0000..}.tar`

## Split

95/5 train/val **by source clip id**. Never split triplets from the same clip across train/val — prevents leakage. Use a deterministic hash of `clip_id` mod 20.

## Script

`src/preprocess.py`

Args:
- `--real-dir`, `--fake-dir`
- `--out-dir` (default `cache/`)
- `--num-workers`
- `--real-triplets-per-clip` (default 6)
- `--fake-triplets-per-clip` (default 3)
- `--crops-per-triplet` (default 2)
- `--resume` (skip shards that already exist)

## Estimated disk

40k real + 40k fake clips × ~4.5 triplets avg × 2 crops × 590 KB ≈ **~280 GB**.

Drop `--crops-per-triplet` to 1 if tight.

## Determinism

Seed the sampler per `clip_id`. Reprocessing the same MP4 twice with the same seed must produce byte-identical shards — this is a verification step.
