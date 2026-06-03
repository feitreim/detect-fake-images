# Image Experiment — Small-Scale AI Image Detection

## Goal

A faster, cheaper port of the video detector to single images: does the
"small model on tiny crops picks up generator artifacts" hypothesis hold for a
text-to-image generator the same way it does for text-to-video?

## Pipeline (mirrors the video one, minus the temporal axis)

| Step | Video                      | Image                                   |
| ---- | -------------------------- | --------------------------------------- |
| Real | OpenVid-1M clips           | COCO 2017 val (5000 photos + captions)  |
| Fake | Seedance/Hunyuan from caps | FLUX.2-klein from captions              |
| Crop | 3-frame 64×64 tubelets     | single-frame 64×64 patches              |
| Feat | 25 stats (incl. temporal)  | 21 stats (temporal dropped)             |
| Model| LR / CNN / ViT             | LR (CNN/ViT reusable later)             |

## Pairing & the codec confound (the two design decisions)

- **Pairing is distributional, not pixel-level.** FLUX is text→image, so a fake
  is not a continuation of a specific real photo. We generate from COCO captions
  so real/fake share *scene types*; since every feature is a single-crop
  statistic, that's enough to control for content.
- **Codec must be matched.** COCO photos are JPEG; raw FLUX output is lossless.
  If left as PNG, the detector becomes a JPEG-vs-clean discriminator, not an AI
  detector. `preprocess_images.py --fake-codec jpeg` re-encodes fakes to JPEG
  q90 (single pass, matching the reals) before cropping. Reals are never
  re-encoded. The `png` condition is the deliberate ablation: the gap between
  `jpeg` and `png` accuracy is the share of signal that was just compression.

## Scripts

- `src/download_real_images.py` — COCO val parquet → download original JPEGs → `real_images_manifest.parquet`.
- `src/generate_fake_images.py` — backend-agnostic (`klein-9b` default / `klein-4b` / `sdxl-turbo`); caption → PNG. **GPU.**
- `src/preprocess_images.py` — `--fake-codec {jpeg,png}` → paired 64×64 crops in `cache_images/{codec}/`.
- `src/features_images.py` — 21 hand-crafted features → npz.
- `src/train_lr.py` — shared with video; reads `feature_names` from the npz.

## Runbook

```bash
# 1. real data (local, ~5000 JPEGs from the COCO CDN)
python -m src.download_real_images

# 2. fakes — ON THE RENTED GPU
pip install "git+https://github.com/huggingface/diffusers.git" transformers accelerate safetensors sentencepiece
python -m src.generate_fake_images --max-images 20      # dry run first
python -m src.generate_fake_images                      # full 5000

# 3+. both codec conditions, then compare LR
python -m src.preprocess_images --fake-codec jpeg --out-dir cache_images/jpeg
python -m src.preprocess_images --fake-codec png  --out-dir cache_images/png
python -m src.features_images --cache-dir cache_images/jpeg --out cache_images/features_jpeg.npz
python -m src.features_images --cache-dir cache_images/png  --out cache_images/features_png.npz
python -m src.train_lr --features cache_images/features_jpeg.npz   # the honest number
python -m src.train_lr --features cache_images/features_png.npz    # ablation upper bound
```

## Non-goals (this round)

- Cross-generator generalization (one target model only, like the video POC).
- Deep models — LR first; the shards keep CNN/ViT on the table if LR underwhelms.
