import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast

from src.model import build_model
from src.dataset import make_loader


def split_params(model):
    """torch.optim.Muon requires ndim == 2 params only. Everything else → AdamW.
    Classifier head and patch_embed projection are kept on AdamW as well."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("head.")
        is_patch_embed = name.startswith("patch_embed.")
        if p.ndim == 2 and not is_head and not is_patch_embed:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def lr_scale(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, max_batches=None):
    model.eval()
    total_loss, n = 0.0, 0
    correct, correct_real, correct_fake = 0, 0, 0
    n_real, n_fake = 0, 0
    wrong_samples = []
    for i, (frames, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        logits = model(frames)
        loss = loss_fn(logits, labels)
        preds = (logits > 0).float()
        hit = (preds == labels)
        correct += hit.sum().item()
        n += labels.numel()
        real_mask = labels == 0
        fake_mask = labels == 1
        correct_real += (hit & real_mask).sum().item()
        correct_fake += (hit & fake_mask).sum().item()
        n_real += real_mask.sum().item()
        n_fake += fake_mask.sum().item()
        total_loss += loss.item() * labels.numel()
        if len(wrong_samples) < 32:
            for j in (~hit).nonzero(as_tuple=False).flatten().tolist():
                wrong_samples.append((int(labels[j].item()), float(logits[j].item())))
                if len(wrong_samples) >= 32:
                    break
    model.train()
    return {
        "loss/val": total_loss / max(1, n),
        "acc/val": correct / max(1, n),
        "acc/val_real": correct_real / max(1, n_real),
        "acc/val_fake": correct_fake / max(1, n_fake),
    }, wrong_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--size", default="small", choices=["tiny", "small", "medium"])
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--adamw-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--limit-shards", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--wandb-project", default="aivideodetector")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    run_name = args.name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    import wandb
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args), mode=args.wandb_mode)

    model = build_model(args.size, crop_size=args.crop_size, patch_size=args.patch_size).to(device)
    muon_params, adamw_params = split_params(model)

    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_muon + n_adamw == n_total, f"split mismatch: {n_muon}+{n_adamw} != {n_total}"
    print(f"params: muon={n_muon:,}  adamw={n_adamw:,}  total={n_total:,}")

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon not found — upgrade PyTorch")
    muon = torch.optim.Muon(
        muon_params,
        lr=args.muon_lr,
        momentum=0.95,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=args.adamw_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16

    train_loader = make_loader(
        args.cache_dir, "train", args.batch_size,
        num_workers=args.num_workers, limit_shards=args.limit_shards,
    )
    val_loader = make_loader(
        args.cache_dir, "val", args.batch_size,
        num_workers=max(1, args.num_workers // 2), limit_shards=args.limit_shards,
    )

    total = args.max_steps or args.total_steps
    step = 0
    best_val = 0.0
    t0 = time.time()
    model.train()

    while step < total:
        for frames, labels in train_loader:
            if step >= total:
                break
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            s = lr_scale(step, args.warmup_steps, args.total_steps)
            for g in muon.param_groups:
                g["lr"] = args.muon_lr * s
            for g in adamw.param_groups:
                g["lr"] = args.adamw_lr * s

            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)

            with autocast(device_type=device if device != "mps" else "cpu", dtype=amp_dtype, enabled=use_amp):
                logits = model(frames)
                loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step()
            adamw.step()

            if step % args.log_every == 0:
                wandb.log({
                    "loss/train": loss.item(),
                    "lr/muon": args.muon_lr * s,
                    "lr/adamw": args.adamw_lr * s,
                    "step_per_s": (step + 1) / (time.time() - t0),
                }, step=step)

            if step > 0 and step % args.eval_every == 0:
                metrics, wrong = evaluate(model, val_loader, device, loss_fn, max_batches=50)
                wandb.log(metrics, step=step)
                print(f"step {step}: {metrics}")
                if metrics["acc/val"] > best_val:
                    best_val = metrics["acc/val"]
                    torch.save(
                        {"model": model.state_dict(), "step": step, "args": vars(args), "best_val": best_val},
                        run_dir / "best.pt",
                    )
                torch.save(
                    {"model": model.state_dict(), "step": step, "args": vars(args)},
                    run_dir / "last.pt",
                )

            step += 1

    torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, run_dir / "last.pt")
    wandb.finish()
    print(f"done. best val acc: {best_val:.4f}")


if __name__ == "__main__":
    main()
