import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class ShardDataset(IterableDataset):
    def __init__(self, shard_dir, shuffle=True, limit_shards=None):
        self.shards = sorted(Path(shard_dir).glob("shard-*.pt"))
        if limit_shards is not None:
            self.shards = self.shards[:limit_shards]
        if not self.shards:
            raise FileNotFoundError(f"no shards in {shard_dir}")
        self.shuffle = shuffle

    def __iter__(self):
        info = get_worker_info()
        shards = list(self.shards)
        if info is not None:
            shards = shards[info.id::info.num_workers]
        if self.shuffle:
            random.shuffle(shards)
        for path in shards:
            data = torch.load(path, map_location="cpu", weights_only=True)
            frames = data["frames"]
            labels = data["labels"]
            idxs = list(range(len(frames)))
            if self.shuffle:
                random.shuffle(idxs)
            for i in idxs:
                yield frames[i], int(labels[i].item())


def make_loader(cache_dir, split, batch_size, num_workers=2, limit_shards=None):
    ds = ShardDataset(Path(cache_dir) / split, shuffle=(split == "train"), limit_shards=limit_shards)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=(split == "train"),
    )
