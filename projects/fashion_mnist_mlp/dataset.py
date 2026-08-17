"""FashionMNIST data plumbing: three loaders instead of two.

The notebooks measured quality straight on the test set. Here the test set is
touched once, at the very end of a run. Every decision taken while training
(how many epochs, which learning rate, which checkpoint to keep) is made on a
validation split carved out of the training data.

Inspect the loaders on their own:
    uv run python -m projects.fashion_mnist_mlp.dataset
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

DATA_ROOT = Path("data")

CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def get_dataloaders(
    batch_size: int = 64,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return the train, validation and test loaders.

    Shapes coming out of every loader:
        images: [batch, 1, 28, 28]  float32 in [0, 1]
        labels: [batch]             int64 in [0, 9]

    The split is driven by its own generator, so the same seed always gives
    the same validation samples no matter what else consumed randomness.
    """
    train_full = datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=ToTensor()
    )

    val_size = int(len(train_full) * val_fraction)
    train_size = len(train_full) - val_size
    train_data, val_data = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def label_counts(dataset: Dataset) -> torch.Tensor:
    """Class histogram of a split, read from the labels without loading images.

    random_split hands back a Subset, so the labels live in the wrapped
    dataset and have to be picked out by the subset indices.
    """
    if isinstance(dataset, Subset):
        targets = dataset.dataset.targets[dataset.indices]
    else:
        targets = dataset.targets

    return torch.bincount(targets, minlength=len(CLASS_NAMES))


def main() -> None:
    loaders = zip(("train", "val", "test"), get_dataloaders())

    for name, loader in loaders:
        images, labels = next(iter(loader))
        print(
            f"{name:>5}  {len(loader.dataset):>5} samples  "
            f"{len(loader):>4} batches  "
            f"images {tuple(images.shape)} {images.dtype}  "
            f"labels {tuple(labels.shape)} {labels.dtype}"
        )

        counts = label_counts(loader.dataset)
        shares = counts / counts.sum()
        print(
            f"       classes {counts.min().item()}-{counts.max().item()} each, "
            f"shares {shares.min().item():.4f}-{shares.max().item():.4f}"
        )


if __name__ == "__main__":
    main()
