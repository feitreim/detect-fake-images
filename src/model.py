import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


# --- CNN ---

class SmallCNN(nn.Module):
    """Tiny 3D CNN for real-vs-fake on (B, T=3, C=3, H, W) crops."""

    def __init__(self, channels=(16, 32, 64), dropout=0.3):
        super().__init__()
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)

        c0, c1, c2 = channels
        self.conv1 = nn.Conv3d(3, c0, kernel_size=(3, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1))
        self.conv2 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c2, 1)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.permute(0, 2, 1, 3, 4)
        x = (x - self.mean) / self.std
        x = F.relu(self.conv1(x)).squeeze(2)  # (B, c0, H, W)
        x = F.relu(self.conv2(x))             # (B, c1, H/2, W/2)
        x = F.relu(self.conv3(x))             # (B, c2, H/4, W/4)
        x = x.mean(dim=(-2, -1))              # (B, c2)
        x = self.drop(x)
        return self.head(x).squeeze(-1)


# --- ViT ---

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


class VideoViT(nn.Module):
    def __init__(self, crop_size=64, patch_size=8, num_frames=3, d_model=384,
                 depth=6, heads=6, mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        assert crop_size % patch_size == 0
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        grid = crop_size // patch_size
        self.num_patches = grid * grid

        self.patch_embed = nn.Conv3d(3, d_model, kernel_size=(num_frames, patch_size, patch_size),
                                     stride=(num_frames, patch_size, patch_size))
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.permute(0, 2, 1, 3, 4)
        x = (x - self.mean) / self.std
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])
        return self.head(x).squeeze(-1)


# --- factory ---

VIT_CONFIGS = {
    "tiny":   dict(d_model=192, depth=4, heads=3, mlp_ratio=4.0),
    "small":  dict(d_model=384, depth=6, heads=6, mlp_ratio=4.0),
    "medium": dict(d_model=512, depth=8, heads=8, mlp_ratio=4.0),
}

CNN_CONFIGS = {
    "cnn-xs":  dict(channels=(8, 16, 32), dropout=0.5),
    "cnn-s":   dict(channels=(16, 32, 64), dropout=0.3),
    "cnn-m":   dict(channels=(32, 64, 128), dropout=0.3),
}


def build_model(size="cnn-s", crop_size=64, patch_size=8, **overrides):
    if size.startswith("cnn"):
        cfg = {**CNN_CONFIGS[size], **overrides}
        return SmallCNN(**cfg)
    cfg = {**VIT_CONFIGS[size], "crop_size": crop_size, "patch_size": patch_size, **overrides}
    return VideoViT(**cfg)
