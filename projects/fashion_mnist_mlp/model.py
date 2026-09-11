"""The classifier itself. This file is yours to write."""

import torch
from torch import nn


class MLP(nn.Module):
    """A plain feed-forward classifier for 28x28 grayscale images.

    Shape flow:
        input:   [batch, 1, 28, 28]
        flatten: [batch, 784]
        hidden:  [batch, hidden]
        logits:  [batch, num_classes]
    """

    def __init__(self, hidden: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        # TODO: nn.Flatten(), then an nn.Sequential of Linear / ReLU layers
        # ending in nn.Linear(hidden, num_classes).
        raise NotImplementedError("build the layers in MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: flatten, then run the stack. Return raw logits — no softmax
        # here, nn.CrossEntropyLoss applies log_softmax internally.
        raise NotImplementedError("write MLP.forward")
