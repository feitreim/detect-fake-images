# Model

Non-factorized 3D tubelet ViT — standard ViViT "Model 1" shape, trimmed to the smallest thing that can work.

## Input

```
x: (B, T=3, C=3, H=256, W=256)  uint8
```

Convert to float, normalize with ImageNet mean/std per-frame.

## Tubelet embedding

Non-overlapping tubelets of shape `(T=3, C=3, H=16, W=16)`.

Patches per clip:
```
(256 / 16) × (256 / 16) = 16 × 16 = 256 patches
```

Each tubelet is `3·3·16·16 = 2304` scalars. Cleanest expression:

```python
self.patch_embed = nn.Conv3d(
    in_channels=3,
    out_channels=d_model,
    kernel_size=(3, 16, 16),
    stride=(3, 16, 16),
)
# (B, d_model, 1, 16, 16) → flatten spatial → (B, 256, d_model)
```

Because `T_kernel == T_input`, the temporal dim collapses to 1 and no temporal positional encoding is needed.

## Positional embedding + CLS

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 257, d_model))  # 256 + CLS
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
```

## Encoder

Pre-norm transformer blocks (standard `LayerNorm → MHA → residual → LayerNorm → MLP → residual`).

Config options in `build_model(size: str)`:

| Size | d_model | depth | heads | mlp_ratio | params |
|---|---|---|---|---|---|
| `small` | 384 | 6 | 6 | 4.0 | ~12M |
| `medium` | 512 | 8 | 8 | 4.0 | ~28M |

Start with `small`.

## Regularization

- Dropout 0.1 in attention + MLP.
- Stochastic depth (DropPath) linearly 0 → 0.1 across layers.

## Head

```python
self.norm = nn.LayerNorm(d_model)
self.head = nn.Linear(d_model, 1)  # single logit, used with BCEWithLogitsLoss
```

## File

`src/model.py`. Keep under ~150 lines. One `VideoViT` class + one `build_model(size)` factory. Tinygrad style — no abstractions beyond what's needed.

## Sanity checks at init

- Print `sum(p.numel() for p in model.parameters())`; assert ~12M for `small`.
- Forward a random `(2, 3, 3, 256, 256)` tensor; assert output shape `(2, 1)`.
