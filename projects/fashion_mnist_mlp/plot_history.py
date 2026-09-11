"""Draw the loss and accuracy curves saved by train.py.

Run from the repository root, after a training run:
    uv run python -m projects.fashion_mnist_mlp.plot_history

The gap between the two loss curves is the thing to look at: while they fall
together the model is still learning, once the validation curve turns back up
the extra epochs are only memorising the training set.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

HISTORY_PATH = Path("models") / "fashion_mnist_mlp_history.json"
PLOT_PATH = Path("models") / "fashion_mnist_mlp_history.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a training run")
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--out", type=Path, default=PLOT_PATH)
    return parser.parse_args()


def plot_run(run: dict, out: Path) -> None:
    """One figure, two panels: the loss pair on the left, accuracy on the right."""
    epochs = [row["epoch"] for row in run["epochs"]]
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(10, 4))

    loss_axes.plot(epochs, [row["train_loss"] for row in run["epochs"]], label="train")
    loss_axes.plot(epochs, [row["val_loss"] for row in run["epochs"]], label="val")
    loss_axes.set_xticks(epochs)
    loss_axes.set_xlabel("epoch")
    loss_axes.set_ylabel("loss")
    loss_axes.legend()

    accuracy_axes.plot(epochs, [row["val_accuracy"] for row in run["epochs"]])
    accuracy_axes.set_xticks(epochs)
    accuracy_axes.set_xlabel("epoch")
    accuracy_axes.set_ylabel("val accuracy")

    args = run["args"]
    figure.suptitle(
        f"lr {args['lr']}  hidden {args['hidden']}  batch {args['batch_size']}"
    )
    figure.tight_layout()
    figure.savefig(out, dpi=120)


def main() -> None:
    args = parse_args()

    if not args.history.exists():
        raise SystemExit(f"no history at {args.history}, run train.py first")

    run = json.loads(args.history.read_text())
    plot_run(run, args.out)

    last = run["epochs"][-1]
    print(
        f"{len(run['epochs'])} epochs, "
        f"last train loss {last['train_loss']:.4f}, "
        f"last val loss {last['val_loss']:.4f}"
    )
    if run["test"] is not None:
        print(f"test accuracy {run['test']['test_accuracy']:.4f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
