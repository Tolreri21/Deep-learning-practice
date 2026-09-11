"""Save a grid of training images, one row per class.

Run from the repository root:
    uv run python -m projects.fashion_mnist_mlp.show_samples --per-class 8

Looking at the data before modelling it is the cheap step everyone skips.
Here it answers one concrete question: which classes look alike to a human?
Whatever confuses the eye is what the confusion table of predict.py will show
later.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from projects.fashion_mnist_mlp.dataset import CLASS_NAMES, get_dataloaders

GRID_PATH = Path("models") / "fashion_mnist_samples.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw FashionMNIST samples")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--out", type=Path, default=GRID_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_samples(
    loader: DataLoader, per_class: int
) -> dict[int, list[torch.Tensor]]:
    """The first `per_class` images of every class, keyed by label."""
    samples: dict[int, list[torch.Tensor]] = {
        label: [] for label in range(len(CLASS_NAMES))
    }

    for images, labels in loader:
        for image, label in zip(images, labels.tolist()):
            if len(samples[label]) < per_class:
                samples[label].append(image)

        if all(len(found) == per_class for found in samples.values()):
            break

    return samples


def plot_grid(
    samples: dict[int, list[torch.Tensor]], per_class: int, out: Path
) -> None:
    """One row per class. Each image is [1, 28, 28] and squeezes to [28, 28]."""
    figure, axes = plt.subplots(
        len(CLASS_NAMES), per_class, figsize=(per_class, len(CLASS_NAMES) * 1.15)
    )

    for row, name in enumerate(CLASS_NAMES):
        for column in range(per_class):
            cell = axes[row][column]
            cell.imshow(samples[row][column].squeeze(), cmap="gray")
            cell.set_xticks([])
            cell.set_yticks([])

        axes[row][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)

    figure.tight_layout()
    figure.savefig(out, dpi=120)


def main() -> None:
    args = parse_args()

    train_loader, _, _ = get_dataloaders(batch_size=256, seed=args.seed)
    samples = collect_samples(train_loader, args.per_class)

    missing = [CLASS_NAMES[label] for label, found in samples.items() if not found]
    if missing:
        raise SystemExit(f"no samples found for {missing}")

    args.out.parent.mkdir(exist_ok=True)
    plot_grid(samples, args.per_class, args.out)
    print(f"{len(CLASS_NAMES)} classes x {args.per_class} images, saved {args.out}")


if __name__ == "__main__":
    main()
