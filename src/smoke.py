"""End-to-end smoke test with no real data.

Builds the model, runs forward/backward on random tensors, verifies the
Muon/AdamW param split, and overfits a single batch to near-zero loss.
Run this before spending GPU money.

    python -m src.smoke
"""
import torch
import torch.nn as nn

from src.model import build_model
from src.train import split_params


def main():
    torch.manual_seed(0)
    device = "cpu"  # Muon may not support mps; keep smoke test portable
    print(f"device: {device}")

    model = build_model("small", crop_size=64, patch_size=8).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {n_total:,}")

    muon_params, adamw_params = split_params(model)
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(f"muon params:  {n_muon:,}")
    print(f"adamw params: {n_adamw:,}")
    assert n_muon + n_adamw == n_total

    muon_ids = {id(p) for p in muon_params}
    for name, p in model.named_parameters():
        if id(p) in muon_ids:
            assert p.ndim == 2, f"muon got non-2D param {name} (ndim={p.ndim})"

    C = model.crop_size
    x = torch.randn(2, 3, 3, C, C, device=device)
    y = model(x)
    print(f"forward ok: input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (2,)

    if not hasattr(torch.optim, "Muon"):
        print("WARN: torch.optim.Muon not available — skipping optimizer test")
        return
    muon = torch.optim.Muon(muon_params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.01)
    adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    B = 8
    x = torch.randn(B, 3, 3, C, C, device=device)
    labels = torch.randint(0, 2, (B,), device=device).float()
    print("overfitting a single batch...")
    model.train()
    for step in range(200):
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, labels)
        loss.backward()
        muon.step()
        adamw.step()
        if step % 20 == 0:
            acc = ((logits > 0).float() == labels).float().mean().item()
            print(f"  step {step:3d}  loss {loss.item():.4f}  acc {acc:.3f}")
    assert loss.item() < 0.1, f"failed to overfit single batch (loss={loss.item()})"
    print("smoke test passed.")


if __name__ == "__main__":
    main()
