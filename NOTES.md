# Notes

Short theory for each notebook in this repo. One theme per notebook. Every theme ends with questions I have to answer before moving on.

Answer them in my own words, without looking at the notebook. If I cannot answer one, I go back and redo that part.

- [1. Tensors](#1-tensors) - `pytorch_basics/tensors.ipynb`
- [2. Datasets and DataLoaders](#2-datasets-and-dataloaders) - `pytorch_basics/datasets_dataloaders.ipynb`
- [3. Transforms](#3-transforms) - `pytorch_basics/transforms.ipynb`
- [4. Build the model](#4-build-the-model) - `pytorch_basics/build_model.ipynb`
- [5. Autograd](#5-autograd) - `pytorch_basics/autograd.ipynb`
- [6. Optimization](#6-optimization) - `pytorch_basics/optimization.ipynb`

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

---

## 3. Transforms

### Why they exist

Data on disk is not in the form a model needs. Images are PIL objects, labels are ints. A model needs float tensors in a fixed layout.

Every `torchvision` dataset takes two callables:

- `transform` changes the features
- `target_transform` changes the label

Both run inside `__getitem__`, so they apply to one sample at a time, never to a batch.

### ToTensor

Does two things at once:

- reorders `[height, width, channels]` into `[channels, height, width]`, the layout PyTorch layers expect
- scales pixels from `uint8` 0-255 to `float32` 0.0-1.0

### Lambda and scatter_

`Lambda` wraps any function into a transform. The usual case is one-hot encoding the label:

```python
Lambda(
    lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
)
```

`scatter_` writes `value` into the positions named by `index`. Start from ten zeros, write 1 at position `y`.

One-hot is not the default. `nn.CrossEntropyLoss` wants the plain int class index, so normal classification leaves `target_transform` out. One-hot is for losses that need a full probability vector.

### Compose and Normalize

`Compose` chains transforms in order, each receiving the output of the previous. `Normalize(mean, std)` does `(x - mean) / std` per channel and must come after `ToTensor`, because it needs a float tensor.

For FashionMNIST the training set gives mean 0.286 and std 0.353. After normalizing, the range is no longer 0.0-1.0 and values go negative. That is correct. The dataset mean moves near 0 and the std near 1, which helps training.

Leakage rule: compute the statistics on the training set only, then reuse the same numbers for validation and test. Recomputing them on the test set leaks information.

### The v2 API

`torchvision.transforms.v2` is the current version and what new code should use. Same names, different import. There `ToTensor` is split in two: `ToImage` builds the tensor, `ToDtype(torch.float32, scale=True)` does the 0-255 to 0.0-1.0 step.

### Questions for theme 3

1. What is the difference between `transform` and `target_transform`? When does each run?
2. `ToTensor` does two separate things. Name both, and say what breaks if each is skipped.
3. Why must `Normalize` come after `ToTensor` and not before?
4. After `Normalize` the pixel values go negative. Is that a bug? Explain.
5. Where do the numbers in `Normalize(mean, std)` come from, and which part of the data must not be used to compute them?
6. I normalize train with train statistics and test with test statistics. What is the name of this mistake, and why does it inflate my score?
7. What does `scatter_` do in the one-hot lambda, and why is the tensor created with `zeros` first?
8. `nn.CrossEntropyLoss` is used for classification. Should the label be one-hot or an int? What happens if I pass the wrong one?
9. Transforms run per sample, not per batch. Why does that matter for augmentation, where two copies of one image should differ?

---

## 4. Build the model

### nn.Module

Every model subclasses `nn.Module` and has two parts:

- `__init__` creates the layers, and must call `super().__init__()` first
- `forward` says how data flows through them

Layers assigned as attributes are registered automatically. That is why their weights appear in `model.parameters()` and move with `.to(device)`.

Call `model(x)`, never `model.forward(x)`. The call operator runs registered hooks around `forward`.

### The layers

- `nn.Flatten` keeps dimension 0, the batch, and flattens the rest. 28 times 28 becomes 784.
- `nn.Linear(in, out)` applies `x @ W.T + b` and touches only the last dimension, so `in_features` must match it. The weight is stored as `[out_features, in_features]`.
- `nn.ReLU` sets negatives to 0 and keeps the rest. The shape does not change.
- `nn.Sequential` is an ordered container, data passes through in the given order.
- `nn.Softmax(dim=1)` turns logits into probabilities. `dim=1` because dimension 0 is the batch, so it runs across classes, not across samples.

Shape flow for a batch of 3 images:

```text
input:    [3, 28, 28]
flatten:  [3, 784]
linear:   [3, 20]
relu:     [3, 20]
linear:   [3, 10]
softmax:  [3, 10]
```

### Logits, probabilities, class

Three different things:

- **logits** are the raw outputs of the last layer, any real number
- **probabilities** come from softmax, in 0.0-1.0 and summing to 1 across classes
- **predicted class** is `argmax`, the index of the largest value

The model returns logits and has no softmax layer. `nn.CrossEntropyLoss` applies log-softmax internally, so putting softmax in the model applies it twice and hurts training.

### Parameters

`model.named_parameters()` gives the name, shape and `requires_grad` of every weight and bias. `requires_grad=True` means autograd will compute a gradient for it. Freezing parameters is how feature extraction works, which is theme 7.

The 784-512-512-10 network has 669706 parameters.

### Questions for theme 4

1. Which two methods does a model need, and what goes in each?
2. What breaks if `super().__init__()` is not called?
3. Why call `model(x)` instead of `model.forward(x)`?
4. Why do the layer weights move to `mps` when I only wrote `model.to(device)`?
5. `nn.Flatten` on `[3, 28, 28]` gives `[3, 784]`, not `[2352]`. Why is the batch dimension kept?
6. `nn.Linear(784, 20)` stores its weight as `[20, 784]`. Why that order and not the other one?
7. Two `Linear` layers with no activation between them. What is the problem?
8. Define logits, probabilities and predicted class. Which one does `nn.CrossEntropyLoss` expect as input?
9. Why is `dim=1` correct in `nn.Softmax(dim=1)`, and what would `dim=0` compute instead?
10. The predictions in this notebook are meaningless. Why?

---

## 5. Autograd

### What it does

During the forward pass PyTorch records every operation. `backward()` walks that record in reverse and applies the chain rule, filling `.grad` on the tensors that asked for it.

Works the same on CPU, MPS and CUDA.

### requires_grad, leaves, grad_fn

- `requires_grad=True` marks a tensor to be optimized. Weights and biases get it, input data does not.
- It spreads forward: any result touching such a tensor also needs gradients.
- A **leaf** is a tensor made by the user, not by an operation. Only leaves with `requires_grad=True` get a populated `.grad`.
- `grad_fn` is the function used to go backward through that step. Leaves have none.

### backward

`loss.backward()` fills `.grad` on every leaf that needs it. The gradient always has the same shape as its tensor, which is a quick sanity check.

The graph is freed after `backward()`. A second call on the same graph raises unless `retain_graph=True`. This is why the graph gets rebuilt every iteration.

`backward()` with no argument works only on a scalar, which is why a loss is always reduced to one number. For a non-scalar output a tensor of the same shape must be passed to say how each element is weighted.

### Gradients accumulate

`backward()` adds to `.grad`, it does not replace it. Three passes without clearing give three times the gradient.

In a real loop the clearing is `optimizer.zero_grad()`. Forgetting it does not crash: the model still trains, but every step uses the sum of all previous gradients, so the updates are wrong and grow over time.

### Turning tracking off

- `torch.no_grad()` for a block, used in evaluation and inference. No graph is built, so it is faster and uses less memory.
- `.detach()` for one tensor, used when a value has to leave the graph, for example to store or plot it.
- `requires_grad_(False)` freezes a parameter so the optimizer cannot change it. That is feature extraction in transfer learning.

### Questions for theme 5

1. Which tensors in a network get `requires_grad=True`, and which never do?
2. What is a leaf tensor? Which tensors actually get a populated `.grad`?
3. What is `grad_fn`, and why is it `None` on `w` but set on `z`?
4. After `loss.backward()`, what shape does `w.grad` have? Why is that a useful check?
5. Calling `backward()` twice raises an error. Why, and what does that tell me about how the graph lives?
6. `backward()` on a non-scalar raises. Why does a loss never hit this problem?
7. What exactly does `optimizer.zero_grad()` prevent? Describe what training looks like without it.
8. Name the difference between `torch.no_grad()` and `.detach()`. When do I reach for each?
9. How do I freeze a layer, and what does the optimizer do with it afterwards?
10. Why is the graph rebuilt on every iteration instead of being reused?

---

## 6. Optimization

### The loop

Training is guess and correct, repeated. The model guesses, the loss says how wrong it was, autograd computes the gradient, the optimizer moves the weights against it.

One pass over the whole training set is an epoch. Each epoch has a train phase and an eval phase.

### Hyperparameters

Set by hand, not learned:

- **learning rate** is the step size. Too small and training crawls, too large and the loss jumps around or diverges.
- **batch size** is how many samples pass before one weight update.
- **epochs** is how many times the whole training set is used.

### Loss function

`nn.CrossEntropyLoss` for single-label classification with more than two classes.

- it takes **logits**, not probabilities, because it applies log-softmax itself
- it takes the target as an **int class index** of shape `[batch]`, not one-hot

Sanity check: an untrained model on 10 classes should start near `ln(10)`, about 2.30. A first loss far from that means something is wired wrong.

### Optimizer

Holds the parameters and the update rule. `model.parameters()` is the link to the model, so anything missing from that list will never change.

SGD is the simplest rule. Adam adapts the step per parameter and usually needs less tuning.

### The three steps

Inside every batch, in this order:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

1. `zero_grad` clears old gradients, because `backward` adds to them rather than replacing them
2. `backward` computes the new gradients
3. `step` applies them to the weights

Order matters. Clearing after `backward` throws away the gradients just computed, and the model never learns.

### train and eval mode

`model.train()` and `model.eval()` switch layers that behave differently in the two phases, dropout and batch norm above all. A plain MLP has neither, so the calls change nothing there, but they are still written, because adding dropout later would silently break evaluation.

`torch.no_grad()` is a separate thing. `eval()` changes layer behaviour, `no_grad()` stops the graph being built. Evaluation wants both.

### Logging and accuracy

`loss.item()` for logging, on purpose. Keeping the tensor would keep its whole graph alive and leak memory across the epoch.

Accuracy compares `pred.argmax(1)` with the target. No softmax needed, because softmax does not change which value is largest.

### Reading the numbers

- train loss should fall steadily. Flat from the start usually means the learning rate is too small, or `zero_grad` and `step` are misplaced.
- test loss should follow it down. Train loss falling while test loss turns up is overfitting.
- accuracy is what gets reported, loss is what gets optimized. They can move apart.

Measured run, SGD at `lr=1e-3`, 5 epochs: train loss 2.2375 down to 1.1589, accuracy 0.5154 up to 0.6402. Slow on purpose, so the trend is visible.

### Questions for theme 6

1. Name the three steps inside a batch, in order. What breaks if `zero_grad` comes last?
2. Why does `nn.CrossEntropyLoss` take logits and not probabilities?
3. What shape and dtype does the target have for `nn.CrossEntropyLoss`? What happens with one-hot instead?
4. An untrained 10-class model starts at loss 2.30. Where does that number come from?
5. What does `model.parameters()` do for the optimizer, and what happens to a tensor left out of it?
6. What is the difference between `model.eval()` and `torch.no_grad()`? Why does evaluation use both?
7. My model has no dropout or batch norm. Is `model.train()` pointless? Explain.
8. Why log `loss.item()` instead of `loss`?
9. Accuracy uses `argmax` with no softmax. Why is that correct?
10. Train loss keeps falling, test loss starts rising. What is happening and what is it called?
11. The learning rate is raised by 100 times and the loss becomes `nan`. What happened?
