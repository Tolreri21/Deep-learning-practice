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

## Notes

`NOTES.md` holds short theory for every notebook, split by theme. Each theme ends with questions that have to be answered before moving on. A new notebook adds a new theme there.

## Data

Datasets download into `data/` on first run and are ignored by git. FashionMNIST is about 30 MB.

## Checks

```bash
uv run ruff check .
uv run ruff format --check
```

CI runs both on every push and pull request. `pre-commit` runs the same plus `nbstripout`, which clears notebook outputs so diffs stay readable.

## Device

The notebooks pick CUDA, then Apple MPS, then CPU. On this machine that is `mps`.
