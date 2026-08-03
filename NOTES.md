# Notes

Short theory for each notebook in this repo. One theme per notebook. Every theme ends with questions I have to answer before moving on.

Answer them in my own words, without looking at the notebook. If I cannot answer one, I go back and redo that part.

- [1. Tensors](#1-tensors) - `pytorch_basics/tensors.ipynb`
- [2. Datasets and DataLoaders](#2-datasets-and-dataloaders) - `pytorch_basics/datasets_dataloaders.ipynb`

---

## 1. Tensors

### What a tensor is

An n-dimensional array, like a NumPy array, with two extra abilities: it can live on a GPU, and it can track gradients. Inputs, outputs, weights and gradients are all tensors.

### Creating them

- `torch.tensor(data)` from a Python list, dtype is inferred
- `torch.from_numpy(arr)` from a NumPy array
- `torch.ones_like(x)`, `torch.rand_like(x)` keep the shape of `x`
- `torch.rand(shape)`, `torch.ones(shape)`, `torch.zeros(shape)` from a shape tuple

### The three attributes to check first

`shape`, `dtype`, `device`. Almost every error at the start is one of these three being wrong.

Common dtype traps:

- a list of ints gives `int64`, a list of floats gives `float32`
- `torch.from_numpy` on a default NumPy array gives `float64`, but layers expect `float32`

### Devices

Tensors start on the CPU and have to be moved. `.to(device)` returns a new tensor, it does not move in place, so the result must be assigned. Both operands of an operation must be on the same device.

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

### Operations

- indexing and slicing work like NumPy
- `torch.cat([a, b], dim=1)` joins along an existing dimension, so that dimension grows
- `@`, `.matmul()`, `torch.matmul(out=)` are matrix multiplication
- `*`, `.mul()`, `torch.mul(out=)` are element-wise
- `.item()` pulls a Python number out of a one-element tensor, used for logging loss
- methods ending in `_` are in-place, they save memory but destroy the old value, so autograd does not always allow them

### NumPy bridge

A CPU tensor and the NumPy array made from it share memory. Changing one changes the other. This works for CPU tensors only, so on `mps` or `cuda` call `.cpu()` first.

### Questions for theme 1

1. What two things can a tensor do that a NumPy array cannot?
2. `torch.tensor([1, 2, 3])` and `torch.tensor([1.0, 2.0, 3.0])` give different dtypes. Which ones, and why does it matter before a model sees the data?
3. Why is `tensor.to(device)` on its own a bug? What is the fix?
4. What is the difference between `a @ b` and `a * b`? What shapes does each need?
5. Shapes `(3, 1)` and `(1, 4)` are added. What is the output shape, and what rule produces it?
6. Why does `loss.item()` appear in training loops instead of just `loss`?
7. What does a trailing underscore mean, as in `add_`, and what is the risk?
8. I change a CPU tensor and its NumPy array changes too. Why? How do I get a copy that does not?
9. `.numpy()` on an `mps` tensor raises an error. Why, and what is the fix?

---

## 2. Datasets and DataLoaders

### The split of responsibility

`Dataset` stores samples and labels and answers two questions: how many are there, and give me sample `i`.

`DataLoader` wraps a `Dataset` and adds batching, shuffling and parallel loading.

They are separate so data code stays out of training code.

### Built-in datasets

`torchvision.datasets` has ready ones such as FashionMNIST: 28x28 grayscale, 10 classes, 60000 train and 10000 test.

`transform=ToTensor()` converts a PIL image to a float tensor and rescales pixels from 0-255 to 0.0-1.0.

Indexing a `Dataset` gives one pair, not a batch: an image `[1, 28, 28]` and a plain int label.

### A custom Dataset

Exactly three methods:

- `__init__` runs once, stores paths, the label table and the transforms
- `__len__` returns the number of samples
- `__getitem__` loads and returns one sample by index

Loading happens in `__getitem__`, not in `__init__`. Reading everything up front breaks as soon as the data is larger than RAM.

`read_image` returns `uint8` in 0-255, unlike `ToTensor()` which returns floats in 0.0-1.0. A transform has to convert it.

### DataLoader

```python
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
```

- `len(dataloader)` is the number of batches, `len(dataloader.dataset)` is the number of samples
- 60000 samples at `batch_size=64` gives 938 batches, the last one smaller
- shuffle the training set so the model does not learn the order, do not shuffle the test set
- on this Mac start with `num_workers=0`, workers add startup cost and can hang in notebooks

The loader stacks samples and adds a batch dimension in front:

```text
one sample:  [1, 28, 28]
one batch:   [64, 1, 28, 28]
labels:      [64]
```

One pass over the loader is one epoch.

### Questions for theme 2

1. What does `Dataset` do and what does `DataLoader` do? Why are they two objects instead of one?
2. Which three methods does a custom `Dataset` need, and what does each return?
3. Why load images in `__getitem__` instead of `__init__`? When does the other choice break?
4. `training_data[0]` gives shape `[1, 28, 28]` but a batch is `[64, 1, 28, 28]`. What does each number mean, and where did the extra one come from?
5. `len(dataloader)` is 938 and `len(dataloader.dataset)` is 60000. Explain both numbers.
6. Why `shuffle=True` for training and `shuffle=False` for testing?
7. `ToTensor()` and `read_image` return different dtypes and ranges. Which are they, and what breaks if I mix them up?
8. What is one epoch in terms of the DataLoader?
9. I have 100 samples and `batch_size=32`. How many batches, and how big is the last one?
