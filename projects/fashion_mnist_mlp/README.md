# FashionMNIST MLP baseline

The first mini-project after the basics notebooks, written as plain scripts.

## Task

Classify 28x28 grayscale clothing images into 10 classes with a fully connected
network. No convolutions yet — this is the baseline a CNN has to beat later.

## Layout

```text
dataset.py       FashionMNIST, train/val split, three DataLoaders
show_samples.py  saves a grid of training images, one row per class
model.py         the MLP
train.py         training loop, validation, checkpointing, one final test pass
predict.py       per class accuracy, the worst mix-ups, single errors
plot_history.py  draws the loss and accuracy curves of a run
utils.py         device selection and seeding
```

## Run

From the repository root:

```bash
uv run python -m projects.fashion_mnist_mlp.dataset          # check the shapes
uv run python -m projects.fashion_mnist_mlp.show_samples     # look at the data
uv run python -m projects.fashion_mnist_mlp.train --epochs 5
uv run python -m projects.fashion_mnist_mlp.predict --errors 10
uv run python -m projects.fashion_mnist_mlp.plot_history
```

Run them as modules from the root, not as file paths — that keeps the imports
between the files working and keeps `data/` and `models/` pointing at the
repository root.

The checkpoint goes to `models/fashion_mnist_mlp.pth` and the per-epoch numbers
to `models/fashion_mnist_mlp_history.json`, so two runs can be compared after the
fact. Neither is tracked by git.

## What is different from the notebooks

Validation is separated from test. `optimization.ipynb` scored the model on the
test set every epoch, which quietly turns the test set into a tuning set. Here
10% of the training data becomes a validation split, the best epoch is chosen by
validation accuracy, and the test set is evaluated once at the end.

## Still to write

- `MLP.__init__` and `MLP.forward` in `model.py`
- `train_one_epoch` and `evaluate` in `train.py`

Everything else — argument parsing, device, seeding, checkpoint save/load,
error inspection — is already in place.
