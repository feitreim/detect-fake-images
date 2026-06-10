# AI Image Detector

My driving idea comes from training GANs: detecting whether an image is the
output of a generator model isn't actually very difficult. When you train a GAN
you have to *balance* the generator and discriminator, and I kept noticing that
a discriminator as large and powerful as the generator is a bad way to achieve
that balance, it wildly outperforms the generator. The discriminator routinely
separates real from fake off of errors that aren't perceptible to humans.

My claim: a model that detects whether outputs come from a specific generator
can be trained with a tiny fraction of the compute that the generator itself
needed. I think small models operating on even small crops of images.

# The Image Experiment

**Task:** real-vs-fake on 3×64×64 image crops. Real = original COCO 2017 photos,
fake = [FLUX.2-klein](https://huggingface.co/black-forest-labs) generations from
those same COCO captions. **40,987 paired images**, aligned by id so a given
scene exists as both a real photo and its synthetic twin.


## Results — non-neural baseline vs. a small CNN

All models see the *identical* crops: 589,504 train / 66,288 val (8 crops each
of 8,286 val images), jpeg-matched. "Crop" columns score each 64×64 crop alone;
"image" columns average the 8 crop scores per image before thresholding.

| Model                            | Params    | Crop Acc  | Crop AUROC | Image Acc | Image AUROC |
| -------------------------------- | --------- | --------- | ---------- | --------- | ----------- |
| Logistic regression (21 feats)   | 25        | 70.0%     | 0.758      | —         | —           |
| Small 2D CNN (raw pixels)        | 93,377    | 87.9%     | 0.950      | 93.8%     | 0.985       |
| **Tiny ViT** (8×8 patches)       | 1,829,761 | **88.6%** | **0.958**  | **95.4%** | **0.991**   |

Both neural nets beat the logistic regression by ~+0.2 AUROC; the ViT edges the CNN
by +0.008. The interesting detail is *why the gap opened up*: the logistic
regression is **saturated** — it scored 0.768 AUROC on an earlier 5k subset and
0.758 here, so 8× more data bought the 25-parameter linear model nothing. Both
neural nets climbed every epoch (CNN 0.814 → 0.950, ViT 0.868 → 0.958) and
**neither had plateaued**, with val tracking train (no overfit). More data only
helps the model with the capacity to absorb it.

### Image-level calls: average the crops

A single 64×64 crop is a noisy estimate of "is this image fake" — different
regions carry different amounts of generator artifact. Averaging the 8 crop
scores per image (free test-time ensembling, no retraining) roughly halves the
error rate: CNN 87.9% → 93.8%, ViT 88.6% → 95.4%. The ViT's image-level
confusion matrix over the 8,286 val images:

```
           pred_real  pred_fake
  real          3945        198      (4.8% of reals flagged)
  fake           181       3962      (4.4% of fakes missed)
```

The 21 hand-crafted features tell you what signal exists: the logistic regression
leans hardest on `edge_R`, `noise_B`, `edge_G` — Sobel edge energy and noise
residuals, the high-frequency fingerprints a recompressed generation leaves behind.
The CNN learns spatially-aware versions of exactly those cues and goes further.

## Pipeline

```
download_real_images   # pull COCO reals by caption/url manifest
generate_fake_images   # FLUX.2-klein from the same captions (backend-agnostic)
preprocess_images      # paired 64x64 crops, --fake-codec {jpeg,png}
                       #   -> cache_images/jpeg_full/{train,val}/shard-*.pt
features_images        # 21 spatial features per crop -> features_*.npz   (for the LR)
train_lr               # logistic-regression baseline
train                  # small CNN baseline (--model vit for a tiny ViT)
                       #   -> runs_images/<model>/{final.pt,metrics.json}
eval                   # re-eval saved checkpoints: crop + image-level metrics
```

Reproduce the comparison above:

```sh
uv run python -m src.preprocess_images \
    --real-manifest data/real_images_manifest_full.parquet \
    --fake-manifest data/fake_images_manifest.parquet \
    --fake-codec jpeg --out-dir cache_images/jpeg_full
uv run python -m src.features_images --cache-dir cache_images/jpeg_full \
    --out cache_images/features_jpeg_full.npz
uv run python -m src.train_lr  --features cache_images/features_jpeg_full.npz
uv run python -m src.train --cache-dir cache_images/jpeg_full --epochs 6
```

The 21 features (`features_images.py`): per-channel pixel mean/std, DCT
high-frequency energy, 8×8 patch-variance stats, gaussian noise-residual
magnitude, and Sobel edge energy.

## Codec-matched preprocessing

A naive setup leaks: COCO reals are JPEG, FLUX fakes are lossless PNG, so a
detector can "win" by learning *PNG vs JPEG* instead of *generator artifacts*.
`preprocess_images.py --fake-codec jpeg` re-encodes every fake to JPEG q90
before cropping, matching the reals' single-pass JPEG. `--fake-codec png` is
there for ablation.

Crops are sampled with a **per-`image_id` seed** and the train/val split is a hash
of `image_id`, so every crop of an image — real *and* fake — lands in the same
split. Verified: **0 images shared between train and val.**
