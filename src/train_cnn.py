"""Neural baseline for real-vs-fake image crops — the deep counterpart to train_lr.py.

train_lr.py fits a 25-param logistic regression on 21 hand-crafted features.
This trains a small CNN (or a tiny ViT) on the raw 3x64x64 pixels of the *same*
crops, so the only thing that changes is "hand-crafted stats" vs "learned features."

Usage:
    python -m src.train_cnn --cache-dir cache_images/jpeg_full              # small CNN
    python -m src.train_cnn --cache-dir cache_images/jpeg_full --model vit  # tiny ViT
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --- data: paired image shards ({"real","fake"} stacks of uint8 (C,H,W)) ---

class ImageShardDataset(IterableDataset):
    def __init__(self, shard_dir, shuffle=True):
        self.shards = sorted(Path(shard_dir).glob("shard-*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"no shards in {shard_dir}")
        self.shuffle = shuffle

    def __iter__(self):
        info = get_worker_info()
        shards = self.shards[info.id::info.num_workers] if info else list(self.shards)
        if self.shuffle:
            import random
            random.shuffle(shards)
        for path in shards:
            data = torch.load(path, map_location="cpu", weights_only=True)
            real, fake = data["real"], data["fake"]
            order = torch.randperm(len(real)) if self.shuffle else torch.arange(len(real))
            for i in order:
                yield real[i], fake[i]


def collate_pairs(batch):
    reals, fakes = zip(*batch)
    B = len(reals)
    x = torch.cat([torch.stack(reals), torch.stack(fakes)], dim=0)
    y = torch.cat([torch.zeros(B), torch.ones(B)])
    perm = torch.randperm(2 * B)
    return x[perm], y[perm]


def make_loader(cache_dir, split, batch_size, num_workers=2):
    ds = ImageShardDataset(Path(cache_dir) / split, shuffle=(split == "train"))
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      persistent_workers=num_workers > 0, drop_last=(split == "train"),
                      collate_fn=collate_pairs)


# --- models ---

class ImageCNN(nn.Module):
    """Tiny 2D CNN: three strided convs -> global avg pool -> logit."""

    def __init__(self, channels=(32, 64, 128), dropout=0.3):
        super().__init__()
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)
        c0, c1, c2 = channels
        self.conv1 = nn.Conv2d(3, c0, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c2, 1)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=(-2, -1))
        return self.head(self.drop(x)).squeeze(-1)


def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep).div_(keep)
    return x * mask


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout, drop_path_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        x = x + drop_path(self.attn(self.norm1(x)), self.drop_path_rate, self.training)
        x = x + drop_path(self.mlp(self.norm2(x)), self.drop_path_rate, self.training)
        return x


class ImageViT(nn.Module):
    """Tiny ViT over 8x8 patches of a 64x64 crop."""

    def __init__(self, crop_size=64, patch_size=8, d_model=192, depth=4, heads=3,
                 mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        grid = crop_size // patch_size
        self.num_patches = grid * grid
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        dpr = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([Block(d_model, heads, mlp_ratio, dropout, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = (x - self.mean) / self.std
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0])).squeeze(-1)


# --- train / eval ---

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.sigmoid(logits).cpu())
        labels.append(y)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    pred = (probs >= 0.5).astype(int)
    return dict(acc=accuracy_score(labels, pred),
                bal=balanced_accuracy_score(labels, pred),
                auc=roc_auc_score(labels, probs),
                cm=confusion_matrix(labels, pred),
                n=len(labels))


def print_confusion(cm):
    print("confusion matrix (rows=actual, cols=predicted):")
    print("           pred_real  pred_fake")
    print(f"  real     {cm[0,0]:>9}  {cm[0,1]:>9}")
    print(f"  fake     {cm[1,0]:>9}  {cm[1,1]:>9}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="cache_images/jpeg_full")
    p.add_argument("--model", default="cnn", choices=["cnn", "vit"])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=256, help="number of PAIRS; actual batch is 2x")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="output dir for weights+metrics (default runs_images/<model>)")
    args = p.parse_args()

    device = pick_device()
    torch.manual_seed(0)
    model = (ImageCNN() if args.model == "cnn" else ImageViT()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={n_params:,} device={device}")

    train_loader = make_loader(args.cache_dir, "train", args.batch_size, args.workers)
    val_loader = make_loader(args.cache_dir, "val", args.batch_size, args.workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        m = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={running/seen:.4f}  "
              f"val_acc={m['acc']:.3f}  val_bal={m['bal']:.3f}  val_auc={m['auc']:.3f}  (n={m['n']})")

    print(f"\nFINAL [{args.model}]  val_acc={m['acc']:.3f}  val_bal={m['bal']:.3f}  val_auc={m['auc']:.3f}")
    print_confusion(m["cm"])

    out = Path(args.out) if args.out else Path("runs_images") / args.model
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), args=vars(args)), out / "final.pt")
    metrics = dict(acc=float(m["acc"]), bal=float(m["bal"]), auc=float(m["auc"]),
                   n=m["n"], confusion=m["cm"].tolist())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved weights + metrics to {out}/")


if __name__ == "__main__":
    main()
