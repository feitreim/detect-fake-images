# Generated (Fake) Video Data

## Goal

For each real clip's caption, generate a single ~30-frame fake clip from HunyuanVideo 1.5, at or above 256×256 so the preprocessor can crop down.

## Why 30 frames

Lets the preprocessor sample `(t, t+k, t+2k)` triplets from different temporal offsets, so the detector doesn't just learn "frame 0 looks like a T2V first-frame artifact."

## Model

Primary: `tencent/HunyuanVideo-1.5` via diffusers. Target output ~512×512 at 24 fps, 30 frames. If the square mode's VRAM is prohibitive, fall back to native 544×960 and crop.

Fallback: `Lightricks/LTX-Video` — switch via `--backend ltx`. Keep the pipeline model-agnostic behind a small `Backend` interface in `src/generate_fake.py`.

## Infra

- Rent one H100 on RunPod (spot) or Lambda.
- Upload `real_manifest.parquet` + `src/generate_fake.py`.
- Resume-safe: skip captions whose `{clip_id}.mp4` already exists in `data/fake/`.
- Log VRAM + per-clip wall time. Abort if throughput drops below an expected floor (sign of leak).
- ~60 clips/hr → 40k clips ≈ 28 GPU-hr ≈ $45 on RunPod spot.

## Output schema

`fake_manifest.parquet`:

```
clip_id          str    # same id as paired real clip
mp4_path         str
caption          str
model_name       str    # "hunyuanvideo-1.5"
model_version    str
seed             int
guidance         float
num_steps        int
num_frames       int
width            int
height           int
```

Everything needed to reproduce a specific fake.

## Script

`src/generate_fake.py`

Args:
- `--real-manifest` (default `data/real_manifest.parquet`)
- `--out-dir` (default `data/fake/`)
- `--backend` (`hunyuan` | `ltx`)
- `--num-frames` (default 30)
- `--resolution` (default 512)
- `--seed-base` (default 0 — per-clip seed is `seed_base + hash(clip_id)`)
- `--max-clips` (for dry runs)

## Intentional non-goals

- Generalizing across multiple generators. Logged as future work.
- Image-to-video (we want pure text-to-video so the detector can't exploit a conditioning leak).
