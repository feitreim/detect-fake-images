# AI Video Detector — Overview

## Goal

Train a small video classifier that distinguishes real camera-captured video from frames produced by one specific open-weight text-to-video generator. Scope is intentionally narrow: detect one target generator well before worrying about cross-model generalization.

## Core hypothesis

A plain non-factorized tubelet-ViT operating on tiny 3-frame 256×256 crops, trained from scratch on paired real/fake data (fake videos generated from the captions of the real ones), will pick up generator-specific artifacts quickly. Three frames is intentional — enough to see motion inconsistency, few enough that training is cheap.

## Key decisions

| Decision           | Choice                                                     | Rationale                                                 |
| ------------------ | ---------------------------------------------------------- | --------------------------------------------------------- |
| Real dataset       | OpenVid-1M curated subset (~30k) + MSR-VTT (~10k)          | Clean captions, CC-BY, manageable disk.                   |
| Gen model          | HunyuanVideo 1.5 (primary), LTX-Video (fallback)           | Photorealistic, Apache 2.0, mature diffusers integration. |
| Generation compute | Rented single H100 on RunPod/Lambda                        | ~60–80 clips/GPU-hr → 40k in ~1 GPU-day (~$45).           |
| Detector training  | Same rented H100; MPS locally for smoke tests              | Small model + small inputs → fits easily.                 |
| Framework          | PyTorch                                                    | As specified.                                             |
| Optimizer          | Muon (2D params) + AdamW (embeddings, norms, biases, head) | Standard Muon split.                                      |
| Target scale       | ~40k real + ~40k fake clips                                | Fits a weekend run.                                       |

## Project layout

```
aivideodetector/
├── plan/
│   ├── 00_overview.md
│   ├── 01_data_real.md
│   ├── 02_data_generated.md
│   ├── 03_preprocessing.md
│   ├── 04_model.md
│   ├── 05_training.md
│   └── 06_eval.md
├── data/                    # raw downloads + generated clips (gitignored)
├── cache/                   # preprocessed tensor shards (gitignored)
├── src/
│   ├── download_real.py
│   ├── generate_fake.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
├── .venv/
├── pyproject.toml
└── README.md
```

## Execution order

1. `src/download_real.py` — 40k real clips. Can run while other scripts are being written.
2. `src/model.py` + `src/train.py` skeleton with a random-tensor dataloader. Smoke-train on MPS to confirm forward/backward + Muon param split.
3. Rent H100, run `src/generate_fake.py` for 40k captions. Rsync clips back.
4. `src/preprocess.py` on both halves → tensor shards.
5. Full training run on rented H100 (~6–10 hours for 20k steps at bs=128).
6. `src/eval.py` on val split + held-out generalization tests.

## Confirmed

- GPU rental for ~1–2 GPU-days ($50–100) is acceptable.
- Logging backend: **wandb**.
- Optimizer: **must use `torch.optim.Muon`** (no external Muon implementation).
