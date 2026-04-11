import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from src.model import build_model
from src.dataset import make_loader


@torch.no_grad()
def run(checkpoint, cache_dir, split, batch_size, device, out_csv):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    model = build_model(
        args_dict["size"],
        crop_size=args_dict.get("crop_size", 64),
        patch_size=args_dict.get("patch_size", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = make_loader(cache_dir, split, batch_size, num_workers=2)
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss, n = 0.0, 0
    correct, correct_real, correct_fake = 0, 0, 0
    n_real, n_fake = 0, 0
    per_k = {}

    rows = []
    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        logits = model(frames)
        loss = loss_fn(logits, labels)
        probs = torch.sigmoid(logits)
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
        for i in range(labels.numel()):
            rows.append({
                "label": int(labels[i].item()),
                "logit": float(logits[i].item()),
                "prob": float(probs[i].item()),
                "correct": int(hit[i].item()),
            })

    metrics = {
        "loss": total_loss / max(1, n),
        "acc": correct / max(1, n),
        "acc_real": correct_real / max(1, n_real),
        "acc_fake": correct_fake / max(1, n_fake),
        "n": n,
    }

    labels_t = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
    probs_t = torch.tensor([r["prob"] for r in rows], dtype=torch.float32)
    metrics["auroc"] = float(binary_auroc(probs_t, labels_t))

    print(metrics)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "logit", "prob", "correct"])
            writer.writeheader()
            writer.writerows(rows)


def binary_auroc(probs, labels):
    order = torch.argsort(probs, descending=True)
    labels = labels[order]
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = torch.cumsum(labels, dim=0)
    fp = torch.cumsum(1 - labels, dim=0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = torch.cat([torch.zeros(1), tpr])
    fpr = torch.cat([torch.zeros(1), fpr])
    return torch.trapz(tpr, fpr).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    run(args.checkpoint, args.cache_dir, args.split, args.batch_size, device, args.out_csv)


if __name__ == "__main__":
    main()
