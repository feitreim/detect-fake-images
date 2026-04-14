import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


class PairedShardDataset(IterableDataset):
    """Yields (real_frames, fake_frames) pairs from paired shards."""

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
            real = data["real_frames"]
            fake = data["fake_frames"]
            idxs = list(range(len(real)))
            if self.shuffle:
                random.shuffle(idxs)
            for i in idxs:
                yield real[i], fake[i]


def collate_pairs(batch):
    """Collate (real, fake) pairs into a single (frames, labels) batch.
    Each pair contributes one real (label=0) and one fake (label=1) example.
    Returned batch is shuffled so real/fake ordering is random."""
    reals, fakes = zip(*batch)
    B = len(reals)
    frames = torch.cat([torch.stack(reals), torch.stack(fakes)], dim=0)
    labels = torch.cat([torch.zeros(B), torch.ones(B)])
    perm = torch.randperm(2 * B)
    return frames[perm], labels[perm]


def make_loader(cache_dir, split, batch_size, num_workers=2, limit_shards=None):
    ds = PairedShardDataset(Path(cache_dir) / split, shuffle=(split == "train"), limit_shards=limit_shards)
    return DataLoader(
        ds,
        batch_size=batch_size,  # number of PAIRS — actual batch is 2x this
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=(split == "train"),
        collate_fn=collate_pairs,
    )
