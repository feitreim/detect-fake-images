# Training

## File

`src/train.py`. One file, no PyTorch Lightning, no Hydra. Argparse for config. Trains `small` by default.

## Muon + AdamW parameter split

Muon applies to hidden 2D matrix weights only. Embeddings, biases, norms, and the classifier head go to AdamW.

```python
muon_params = []
adamw_params = []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name in ("cls_token", "pos_embed") or name.startswith("head."):
        adamw_params.append(p)
    elif p.ndim >= 2:
        muon_params.append(p)
    else:
        adamw_params.append(p)
```

Explicitly exclude `cls_token`, `pos_embed`, and the final `Linear` head even though the head is 2D+. This matches the Muon convention.

## Muon source

**Use `torch.optim.Muon`.** If the installed PyTorch version doesn't expose it, upgrade PyTorch until it does — no external Muon implementation.

## Hyperparameters

```
Muon:
  lr            0.02
  momentum      0.95
  nesterov      True
  ns_steps      5

AdamW:
  lr            3e-4
  betas         (0.9, 0.95)
  weight_decay  0.01

Schedule:
  warmup_steps    1000  (linear)
  total_steps    20000  (cosine decay to 10%)

Batch size      128  (fits easily on H100 at small size)
AMP             bf16 on H100, fp16 on MPS
Grad clip       1.0  (global norm)
```

## Logging

wandb. Project name `aivideodetector`, run name defaults to a timestamp. Log:
- `loss/train`, `loss/val`
- `acc/train`, `acc/val`, `acc/val_real`, `acc/val_fake`
- `lr/muon`, `lr/adamw`
- A handful of misclassified `clip_id`s per eval step as a wandb Table, and also dumped to `runs/{name}/misclassified_{step}.txt` for offline inspection.

## Checkpointing

- `runs/{name}/best.pt` — best val accuracy so far.
- `runs/{name}/last.pt` — most recent.

## Smoke test path

```
python -m src.train --size small --max-steps 50 --device mps --cache-subset 100
```

Must run on the laptop against a 100-clip cache subset before any rented-GPU spend. Confirms:
- Forward/backward works.
- Muon param split doesn't crash.
- Loss decreases over 50 steps.
