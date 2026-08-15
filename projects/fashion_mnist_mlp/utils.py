"""Small helpers shared by the scripts of this project."""

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """CUDA if present, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch so a run can be repeated.

    This does not make a run bit-for-bit reproducible on MPS, but it does fix
    the weight initialisation, the train/val split and the shuffling order.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
