"""Dataset: Fashion-MNIST, fetched from its real public source.

Why this dataset. The pipeline needs a *real* public dataset that (a) downloads without
credentials, (b) is large enough that training is a genuine step rather than a
formality, and (c) is a classification task whose errors are interpretable, because
"the model got worse" has to be diagnosable in CI. Fashion-MNIST is 70,000 real 28x28
greyscale images across 10 clothing classes, and unlike MNIST it is not saturated -
a small CNN lands around 91-93%, so a regression in the pipeline actually shows up in
the metric instead of hiding under 99.5%.

Why not CIFAR-10: 3-channel 32x32 is a longer train for the same pipeline lesson, and
this project is about the pipeline, not about squeezing the model.

The split matters more than it looks. A model selected on the same data it is
early-stopped on has a validation score that is optimistic by construction, so there
are three splits, cut with a fixed seed, and the test set is touched exactly once - in
evaluate.py, after training is finished.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

DATA_DIR = Path(os.environ.get("FORGE_DATA_DIR", "data"))
SEED = int(os.environ.get("FORGE_SEED", "20260910"))

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Computed once from the training split only. Using statistics from the full dataset
# would leak test-set information into every normalised training batch - a small leak,
# but a real one, and the kind CI can never detect afterwards.
TRAIN_MEAN, TRAIN_STD = 0.2860, 0.3530


@dataclass
class Splits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    fingerprint: str            # identifies exactly which data produced a model

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _to_tensors(ds) -> Tuple[torch.Tensor, torch.Tensor]:
    """torchvision datasets are PIL-backed and slow to iterate. Materialising once into
    a single normalised float tensor makes every epoch a memory read instead of 60,000
    PIL decodes, which is the difference between a 20-second epoch and a 3-second one."""
    x = ds.data.numpy().astype(np.float32) / 255.0
    x = (x - TRAIN_MEAN) / TRAIN_STD
    x = torch.from_numpy(x).unsqueeze(1)          # N,1,28,28
    y = ds.targets.clone().detach().long()
    return x, y


def load_splits(val_fraction: float = 0.1, download: bool = True) -> Splits:
    from torchvision import datasets            # imported here so `import data` is cheap

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = datasets.FashionMNIST(str(DATA_DIR), train=True, download=download)
    test_raw = datasets.FashionMNIST(str(DATA_DIR), train=False, download=download)

    xtr, ytr = _to_tensors(train_raw)
    xte, yte = _to_tensors(test_raw)

    # Deterministic split. A different validation set per run would make two CI runs
    # incomparable, and "did this commit help?" is the question the pipeline exists
    # to answer.
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(xtr), generator=g)
    n_val = int(len(xtr) * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    fingerprint = hashlib.sha256(
        f"{len(xtr)}:{len(xte)}:{SEED}:{val_fraction}".encode()
        + xtr[train_idx[:64]].numpy().tobytes()
    ).hexdigest()[:16]

    return Splits(
        train=TensorDataset(xtr[train_idx], ytr[train_idx]),
        val=TensorDataset(xtr[val_idx], ytr[val_idx]),
        test=TensorDataset(xte, yte),
        fingerprint=fingerprint,
    )


def loaders(splits: Splits, batch_size: int = 128) -> Dict[str, DataLoader]:
    # num_workers=0 deliberately: the tensors are already in memory, so worker
    # processes would add IPC and pickling cost to move data that is already there.
    common = dict(num_workers=0, pin_memory=False)
    g = torch.Generator().manual_seed(SEED)
    return {
        "train": DataLoader(splits.train, batch_size=batch_size, shuffle=True,
                            generator=g, drop_last=True, **common),
        "val": DataLoader(splits.val, batch_size=512, shuffle=False, **common),
        "test": DataLoader(splits.test, batch_size=512, shuffle=False, **common),
    }


def class_balance(splits: Splits) -> Dict[str, Dict[str, int]]:
    """Reported because accuracy is only a meaningful headline on a balanced problem.
    Fashion-MNIST is exactly balanced; saying so is what earns the right to quote
    accuracy rather than macro-F1."""
    out = {}
    for name, ds in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        _, y = ds.tensors
        counts = torch.bincount(y, minlength=len(CLASSES)).tolist()
        out[name] = {CLASSES[i]: c for i, c in enumerate(counts)}
    return out


if __name__ == "__main__":
    s = load_splits()
    print(json.dumps({"sizes": s.sizes(), "fingerprint": s.fingerprint,
                      "balance_train": class_balance(s)["train"]}, indent=2))
