"""The model. A small CNN, and the reasons it is small are deliberate.

This project is about the pipeline, and the pipeline's claim is that it is
*model-agnostic*. That claim is only testable if the model is a real neural network
whose weights actually have to be trained, exported, versioned and served - not if it
is a stand-in that returns a constant.

So: a genuine convolutional net (two conv blocks, batch norm, dropout, ~93% on
Fashion-MNIST), but deliberately under 500k parameters so that a full training run
finishes inside a CI job. A ResNet-50 would prove nothing extra about the pipeline and
would make the pipeline untestable, which is the opposite of the point.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForgeCNN(nn.Module):
    """Conv-BN-ReLU x2 -> pool, twice, then a small classifier head.

    Batch norm before the activation, not after: it normalises the pre-activation
    distribution, which is what the original formulation does and what keeps the
    ReLU from being fed an already-shifted input.

    Dropout only in the head. Dropout between convolutions fights batch norm - the
    two disagree about the variance the next layer should expect - and on a dataset
    this size the convolutional features are not where the overfitting is.
    """

    def __init__(self, num_classes: int = 10, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),      # bias is redundant
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),       # before BN's own shift
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # 28 -> 14
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 2 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming for ReLU nets. The PyTorch default is Kaiming *uniform* with
        a=sqrt(5), which is tuned for leaky-ReLU and leaves ReLU layers with slightly
        too little variance; fan_out normal is the initialisation the architecture
        actually calls for."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block2(self.block1(x)))


def build(config: Dict | None = None) -> ForgeCNN:
    config = config or {}
    return ForgeCNN(num_classes=config.get("num_classes", 10),
                    width=config.get("width", 32),
                    dropout=config.get("dropout", 0.3))


def parameter_count(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def pick_device() -> torch.device:
    """MPS on Apple silicon, CUDA where present, CPU otherwise.

    CI runs on CPU and a laptop runs on MPS, so the pipeline must not assume either.
    Anything that depends on the device (batch size, worker count) is read from
    config, never hard-coded to whatever the author's machine had.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
