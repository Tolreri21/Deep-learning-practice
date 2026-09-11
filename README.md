# Deep-learning-practice

A structured PyTorch learning lab with practical mini-projects in deep learning, computer vision, and NLP.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and is pinned to Python 3.12.

```bash
uv sync
```

That creates `.venv` and installs everything, including the dev group with `ruff` and `ipykernel`.

To run a notebook, pick the `.venv` interpreter as the kernel in PyCharm or Jupyter.

## Layout

```text
pytorch_basics/   notebooks, one per theme
projects/         mini-projects as scripts, one folder each
models/           saved checkpoints, not tracked
data/             downloaded datasets, not tracked
NOTES.md          theory for each theme plus questions to answer
```

## Notebooks

| Theme | Notebook | Topic |
| --- | --- | --- |
| 1 | `pytorch_basics/tensors.ipynb` | creating tensors, shape, dtype, device, operations, NumPy bridge |
| 2 | `pytorch_basics/datasets_dataloaders.ipynb` | built-in datasets, a custom `Dataset`, batching with `DataLoader` |
| 3 | `pytorch_basics/transforms.ipynb` | `transform` and `target_transform`, `ToTensor`, `Compose`, `Normalize` |
| 4 | `pytorch_basics/build_model.ipynb` | `nn.Module`, layers, shape flow, logits vs probabilities |
| 5 | `pytorch_basics/autograd.ipynb` | `requires_grad`, `backward`, accumulation, `no_grad` and `detach` |
| 6 | `pytorch_basics/optimization.ipynb` | loss, optimizer, the training loop, `train` and `eval` mode |
| 7 | `pytorch_basics/save_load.ipynb` | `state_dict`, `weights_only`, `map_location`, checkpoints |

## Projects

Notebooks are for learning a theme. Projects are scripts, run as modules from
the repository root.

| Project | Folder | Task |
| --- | --- | --- |
| 1 | `projects/fashion_mnist_mlp/` | FashionMNIST baseline: MLP, validation split, checkpoint, error inspection |

```bash
uv run python -m projects.fashion_mnist_mlp.train --epochs 5
```

Each project folder has its own `README.md` with the task and the commands.

## Notes

`NOTES.md` holds short theory for every notebook, split by theme. Each theme ends with questions that have to be answered before moving on. A new notebook or project adds a new theme there.

## Data

Datasets download into `data/` on first run and are ignored by git. FashionMNIST is about 30 MB.

## Checks

```bash
uv run ruff check .
uv run ruff format --check
```

CI runs both on every push and pull request, plus a check that every module under `projects/` imports.

`pre-commit` runs the same linters plus `nbstripout`, which clears notebook outputs so diffs stay readable. It is a dev dependency, but the git hook itself is per clone and has to be installed once:

```bash
uv run pre-commit install
```

## Device

The notebooks pick CUDA, then Apple MPS, then CPU. On this machine that is `mps`.
