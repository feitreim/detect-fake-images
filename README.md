# AI Video Detector

Binary classifier that distinguishes real camera-captured video from AI-generated video. Built around the hypothesis that generator artifacts are detectable through local patch statistics — noise residuals, edge structure, and frequency-domain patterns — even at small crop sizes.

## Results

Proof-of-concept on 100 paired clips (real from OpenVid-1M, fake from Seedance 2.0 image-to-video):

| Model | Train Acc | Val Acc | Val Balanced Acc | Val AUROC | Params |
|---|---|---|---|---|---|
| ViT (tiny) | 100% | 50% | 50% | — | 1.9M |
| CNN (xs) | 100% | 50% | 50% | — | 6.5k |
| **Logistic regression on hand-crafted features** | **67%** | **75%** | **75%** | **0.84** | **25** |

Neural networks memorized clip identity rather than learning real-vs-fake artifacts. The logistic regression model — with zero capacity to memorize — achieved 75% balanced accuracy and 0.84 AUROC on completely held-out clips, confirming that the signal exists in local statistics.

### Top discriminative features

| Feature | |coef|| Description |
|---|---|---|
| noise_R | 4.34 | Noise residual magnitude (red channel) |
| noise_G | 4.13 | Noise residual magnitude (green channel) |
| edge_R | 3.94 | Sobel edge energy (red channel) |
| noise_B | 3.12 | Noise residual magnitude (blue channel) |
| edge_G | 2.69 | Sobel edge energy (green channel) |

Noise residuals (gaussian blur, subtract, measure deviation) and edge energy are the strongest discriminators. Seedance 2.0 produces subtly different noise and edge patterns than real camera sensors.

## Architecture

**Data pipeline:** Real video clips from OpenVid-1M are paired with Seedance 2.0 image-to-video generations conditioned on the first frame. Both real and fake are decoded, temporally sampled into 3-frame triplets, and randomly cropped to 64x64. All crops from a clip go to the same train/val split (no leakage).

**Feature extraction:** 25 hand-crafted features per crop:
- Per-channel pixel mean/std (6)
- Temporal frame-diff statistics (4)
- DCT high-frequency energy ratio per channel (3)
- 8x8 patch variance mean/std per channel (6)
- Noise residual magnitude per channel (3)
- Sobel edge energy per channel (3)

**Classifier:** Logistic regression with L2 regularization (sklearn). Best val performance at C=0.01.

The repo also includes a ViT and CNN for when more data is available.

## Usage

### Setup

```bash
cd aivideodetector
uv venv --python 3.12 && uv sync
```

### Download real videos

```bash
.venv/bin/python -m src.download_real --parts 0
```

### Generate fakes (requires `FAL_KEY`)

```bash
pip install fal-client Pillow
export FAL_KEY=...
.venv/bin/python -m src.generate_fake --max-clips 10  # test
.venv/bin/python -m src.generate_fake                  # full
```

### Preprocess (paired)

```bash
.venv/bin/python -m src.preprocess \
  --real-manifest data/real_manifest.parquet \
  --fake-manifest data/fake_manifest.parquet \
  --crop-size 64 --resize-short 80
```

### Train (logistic regression)

```bash
.venv/bin/python -m src.features          # extract -> cache/features.npz
.venv/bin/python -m src.train_lr          # train + eval
```

### Train (neural network — needs more data)

```bash
.venv/bin/python -m src.train --size cnn-xs --device mps --wandb-mode disabled
.venv/bin/python -m src.train --size tiny --device cuda  # ViT on GPU
```

## Future work

### More data

The single biggest limitation. With 100 paired clips, neural networks memorize clip identity instead of learning generalizable artifacts. Priority next steps:

- **Scale to 1,000+ paired clips.** This likely crosses the threshold where small CNNs and ViTs can generalize. The existing pipeline (`generate_fake.py`) supports resume and concurrent workers — it just needs API budget.
- **Multiple generators.** Currently trained only against Seedance 2.0. Cross-generator evaluation (LTX-Video, HunyuanVideo, etc.) would test how transferable the features are. The feature-engineering approach may transfer better than a learned model since noise/edge patterns are generator-agnostic.
- **Broader real video sources.** OpenVid part 0 is dominated by CelebV face clips (96%), which hit Seedance's content policy filter. Downloading additional parts would diversify scene types.

### Better features

- Learned frequency filters (trainable DCT basis or wavelet coefficients)
- Cross-frame consistency features (optical flow statistics, flickering patterns)
- Multi-scale analysis (same features at 32x32, 64x64, 128x128)

### Model

- Once data allows: train the CNN/ViT architectures already in the repo
- Contrastive learning on paired crops (more data-efficient than classification)
- Ensemble: combine logistic regression features with a small learned model

## Project structure

```
src/
  download_real.py   # OpenVid-1M download via HuggingFace
  generate_fake.py   # Seedance 2.0 via fal.ai API
  preprocess.py      # Paired video -> tensor shards
  features.py        # Hand-crafted feature extraction
  train_lr.py        # Logistic regression training
  model.py           # VideoViT + SmallCNN architectures
  dataset.py         # Paired shard dataloader
  train.py           # Neural network training loop
  eval.py            # Checkpoint evaluation
  smoke.py           # Model sanity check
plan/                # Design docs
```
