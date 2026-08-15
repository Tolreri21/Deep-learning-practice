"""Load the saved checkpoint and look at what the model gets wrong.

Run from the repository root, after train.py has produced a checkpoint:
    uv run python -m projects.fashion_mnist_mlp.predict --errors 10
"""

import argparse
from pathlib import Path

import torch

from projects.fashion_mnist_mlp.dataset import CLASS_NAMES, get_dataloaders
from projects.fashion_mnist_mlp.model import MLP
from projects.fashion_mnist_mlp.utils import get_device

CHECKPOINT_PATH = Path("models") / "fashion_mnist_mlp.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FashionMNIST errors")
    parser.add_argument("--errors", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(device: torch.device) -> MLP:
    """Rebuild the architecture, then pour the saved weights into it."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)

    model = MLP(hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(
        f"loaded epoch {checkpoint['epoch']}, "
        f"val accuracy {checkpoint['val_accuracy']:.4f}"
    )
    return model


def main() -> None:
    args = parse_args()

    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"no checkpoint at {CHECKPOINT_PATH}, run train.py first")

    device = get_device()
    print(f"device: {device}")

    model = load_model(device)
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, seed=args.seed)

    shown = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            probabilities = model(images).softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            wrong = (predicted != labels).nonzero(as_tuple=True)[0]

            for i in wrong.tolist():
                print(
                    f"true {CLASS_NAMES[int(labels[i])]:<12}  "
                    f"predicted {CLASS_NAMES[int(predicted[i])]:<12}  "
                    f"confidence {confidence[i].item():.2f}"
                )
                shown += 1
                if shown == args.errors:
                    return


if __name__ == "__main__":
    main()
