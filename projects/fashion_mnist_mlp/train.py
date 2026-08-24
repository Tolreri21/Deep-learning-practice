"""Train the MLP baseline on FashionMNIST.

Run from the repository root:
    uv run python -m projects.fashion_mnist_mlp.train --epochs 5

Model selection happens on the validation split. The test set is evaluated
once, at the end, with the best checkpoint reloaded.

The two functions that carry the learning are left empty on purpose.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from projects.fashion_mnist_mlp.dataset import get_dataloaders
from projects.fashion_mnist_mlp.model import MLP
from projects.fashion_mnist_mlp.utils import get_device, set_seed

CHECKPOINT_PATH = Path("models") / "fashion_mnist_mlp.pth"
HISTORY_PATH = Path("models") / "fashion_mnist_mlp_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FashionMNIST MLP baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over the training data. Returns the mean loss per sample.

    Your part. Per batch:
        1. move X and y to the device
        2. forward pass -> logits [batch, 10]
        3. loss = loss_fn(logits, y)
        4. optimizer.zero_grad(), loss.backward(), optimizer.step()
        5. accumulate loss.item() * X.size(0)

    model.train() goes before the loop, not inside it.
    """
    raise NotImplementedError("write train_one_epoch")


def evaluate(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Loss and accuracy on a loader. Returns (mean loss per sample, accuracy).

    Your part. model.eval() plus a torch.no_grad() block, no optimizer here.
    The predicted class is logits.argmax(dim=1); count the matches against y.
    """
    raise NotImplementedError("write evaluate")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    hidden: int,
) -> None:
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "hidden": hidden,
        },
        CHECKPOINT_PATH,
    )


def save_history(
    history: list[dict[str, float]],
    args: argparse.Namespace,
    test: dict[str, float] | None = None,
) -> None:
    """Dump the epoch table next to the checkpoint so runs can be compared.

    Written after every epoch, not once at the end, so an interrupted run
    still leaves its numbers behind.
    """
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    run = {"args": vars(args), "epochs": history, "test": test}
    HISTORY_PATH.write_text(json.dumps(run, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(
        f"train {len(train_loader.dataset)}  "
        f"val {len(val_loader.dataset)}  "
        f"test {len(test_loader.dataset)}"
    )

    model = MLP(hidden=args.hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    best_val_accuracy = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_accuracy = evaluate(val_loader, model, loss_fn, device)
        print(
            f"epoch {epoch:>2d}  train loss {train_loss:.4f}  "
            f"val loss {val_loss:.4f}  val acc {val_accuracy:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        save_history(history, args)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, val_accuracy, args.hidden)
            print(f"  saved {CHECKPOINT_PATH}")

    print(f"best val accuracy {best_val_accuracy:.4f}")

    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"no checkpoint written after {args.epochs} epochs")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy = evaluate(test_loader, model, loss_fn, device)
    save_history(
        history, args, {"test_loss": test_loss, "test_accuracy": test_accuracy}
    )

    print(
        f"test loss {test_loss:.4f}  test accuracy {test_accuracy:.4f}  "
        f"(from epoch {checkpoint['epoch']})"
    )
    print(f"history in {HISTORY_PATH}")


if __name__ == "__main__":
    main()
