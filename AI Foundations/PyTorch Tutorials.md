# Senior AI Architect — Practical PyTorch Tutorial

A practical, interview-oriented PyTorch guide covering the fundamentals and implementation of CNNs, RNNs/LSTMs, Transformers, and LoRA.

> This guide is written for someone new to PyTorch. Every code block is followed by **Expected Output** (what you'd actually see if you ran it), **Key Terms** (plain-English definitions), and, where relevant, **Math Behind It** (the formula in words, not just symbols).

## 1. PyTorch Mental Model

PyTorch revolves around four ideas:

1. **Tensor** — multidimensional data that can run on CPU/GPU and participate in autograd.
2. **`nn.Module`** — base class for neural-network layers/models.
3. **Autograd** — automatically computes gradients.
4. **Optimizer** — updates trainable parameters.

Typical training flow:

```text
data → model(x) → loss → loss.backward() → optimizer.step()
```

**Key Terms:**
- **Tensor** — think of it as a NumPy array that PyTorch can also track gradients for and move to a GPU. It's the single data structure PyTorch is built around.
- **`nn.Module`** — the base "blueprint" class every layer and model inherits from. It knows how to track its own trainable weights.
- **Autograd** — short for "automatic differentiation." It records every operation you perform on tensors so it can later compute gradients (derivatives) automatically, without you doing calculus by hand.
- **Optimizer** — an algorithm (like AdamW) that nudges the model's weights in the direction that reduces the loss, using the gradients autograd computed.
- **Loss** — a single number measuring how wrong the model's prediction was. Training = repeatedly making this number smaller.

**Math Behind It:**
Training is essentially repeated **gradient descent**: at each step you compute how much the loss would change if each weight changed slightly (the gradient), then move each weight a small amount in the opposite direction of its gradient, since that's the direction that decreases the loss fastest.

## 2. Tensors

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

print(x)
print(x.shape)
print(x.dtype)

zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
random = torch.randn(2, 3)
```

**Expected Output:**

```text
tensor([1., 2., 3.])
torch.Size([3])
torch.float32
```

`zeros`, `ones`, and `random` aren't printed here, but if you did print them you'd see a 2×3 grid each, e.g. `zeros` prints as `tensor([[0., 0., 0.], [0., 0., 0.]])`. `random` will show different numbers every time you run it, since it's sampled from a normal (Gaussian) distribution.

Useful shape operations:

```python
x = torch.arange(12)

y = x.reshape(3, 4)
z = y.view(2, 6)

a = torch.tensor([1, 2, 3])

print(a.unsqueeze(0).shape)  # [1, 3]
print(a.unsqueeze(1).shape)  # [3, 1]

b = torch.randn(1, 3, 1)
print(b.squeeze().shape)      # [3]
```

**Expected Output:**

```text
torch.Size([1, 3])
torch.Size([3, 1])
torch.Size([3])
```

For reference, `x` is `tensor([0, 1, 2, ..., 11])`, `y` reshapes it into 3 rows of 4, and `z` reshapes those same 12 numbers into 2 rows of 6 — the underlying data doesn't change, only how it's "viewed."

**Key Terms:**
- **Shape** — the size of a tensor along each dimension, e.g. `[2, 3]` means 2 rows, 3 columns. Getting shapes right is the single most common source of PyTorch bugs.
- **`dtype`** — the data type stored in the tensor (e.g. `float32`, `int64`). `torch.tensor([1.0, ...])` defaults to `float32` because the values have decimal points.
- **`reshape` vs `view`** — both change a tensor's shape without changing its data. `view` requires the tensor's memory to be laid out contiguously; `reshape` will copy the data if needed, so it always works.
- **`unsqueeze(dim)`** — inserts a new dimension of size 1 at position `dim`. Commonly used to add a "batch" dimension to a single example.
- **`squeeze()`** — removes all dimensions of size 1. `[1, 3, 1]` becomes `[3]`.

**Math Behind It:**
A tensor of shape `[2, 3]` holds `2 × 3 = 6` numbers total. Reshaping only regroups those 6 numbers into a different grid — the total element count (`numel()`) must stay the same before and after.

## 3. Tensor Operations

```python
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print(a + b)   # element-wise addition
print(a * b)   # element-wise multiplication
print(a @ b)   # matrix multiplication
print(torch.matmul(a, b))
```

**Expected Output:**

```text
tensor([[ 6.,  8.],
        [10., 12.]])
tensor([[ 5., 12.],
        [21., 32.]])
tensor([[19., 22.],
        [43., 50.]])
tensor([[19., 22.],
        [43., 50.]])
```

Reductions:

```python
x = torch.randn(3, 4)

print(x.sum())
print(x.mean())
print(x.max())
print(x.mean(dim=0))
print(x.mean(dim=1))
```

**Expected Output:** (exact numbers will differ every run, since `x` is random — but the *shapes* below are always the same)

```text
tensor(-1.2345)          # a single scalar: sum of all 12 values
tensor(-0.1029)          # a single scalar: mean of all 12 values
tensor(2.1198)           # a single scalar: the largest value
tensor([ 0.31, -0.55, 0.02, -0.12])   # shape [4], mean of each column
tensor([-0.08, 0.15, -0.44])          # shape [3], mean of each row
```

**Key Terms:**
- **Element-wise operation** (`+`, `*`) — applies the operation to each matching position independently. Requires both tensors to have the same (or broadcastable) shape.
- **Matrix multiplication** (`@` / `torch.matmul`) — combines rows of the first matrix with columns of the second using dot products. This is *not* the same as `*`.
- **`dim` argument** — tells a reduction which axis to collapse. `dim=0` collapses rows (reduces down each column), `dim=1` collapses columns (reduces across each row).

**Math Behind It:**
For matrix multiplication `a @ b`, each output entry is a **dot product**: `result[i][j] = Σ a[i][k] * b[k][j]`. For example, the top-left entry above is `1×5 + 2×7 = 19`. Element-wise multiplication, by contrast, is just `a[i][j] * b[i][j]` — no summing, no combining rows with columns.

## 4. Broadcasting

```python
x = torch.randn(3, 4)
bias = torch.randn(4)

y = x + bias
print(y.shape)
```

**Expected Output:**

```text
torch.Size([3, 4])
```

PyTorch broadcasts compatible dimensions automatically.

**Key Terms:**
- **Broadcasting** — PyTorch's rule for combining tensors of different shapes without you manually copying data. It compares shapes from the *right*; dimensions match if they're equal, or if one of them is `1` (or missing).

**Math Behind It:**
Here `x` is `[3, 4]` and `bias` is `[4]`. PyTorch treats `bias` as if it had shape `[1, 4]`, then conceptually repeats it 3 times to match `x`'s shape, so every row of `x` gets the same `bias` vector added to it. No memory is actually copied — this is done efficiently under the hood.

## 5. CPU and GPU

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

x = torch.randn(3, 4).to(device)
```

**Expected Output:** (no printed output — `device` will simply hold `device(type='cpu')` on a machine without an NVIDIA GPU, or `device(type='cuda')` on one with a working CUDA setup)

Move a model:

```python
model = model.to(device)
```

**Key Terms:**
- **`torch.device`** — describes *where* a tensor's memory lives and where its computations run: `"cpu"` or `"cuda"` (NVIDIA GPU). Apple Silicon uses `"mps"` instead.
- **`.to(device)`** — copies a tensor or model's parameters onto that device. Two tensors must be on the *same* device to interact — a common beginner error is `RuntimeError: Expected all tensors to be on the same device`.

## 6. Autograd

```python
x = torch.tensor(3.0, requires_grad=True)

y = x**2 + 2 * x
y.backward()

print(x.grad)
```

**Expected Output:**

```text
tensor(8.)
```

For `y = x² + 2x`, the derivative is `2x + 2`, so at `x=3` the gradient is `8`.

Disable gradient tracking during inference:

```python
with torch.inference_mode():
    prediction = model(x)
```

**Key Terms:**
- **`requires_grad=True`** — tells PyTorch "track every operation on this tensor, because I'll want its gradient later." Model parameters have this set automatically.
- **Computational graph** — as you compute `y = x**2 + 2*x`, PyTorch silently builds a graph of operations behind the scenes. `backward()` walks this graph in reverse to compute gradients.
- **`.backward()`** — triggers the gradient computation, using the **chain rule** from calculus, and stores the result in `.grad` on every leaf tensor that required it.
- **`.grad`** — where the computed gradient (derivative) is stored after `backward()` runs. It accumulates by default, which is why `optimizer.zero_grad()` exists (see Section 14).
- **`inference_mode()` / `no_grad()`** — context managers that turn off gradient tracking, since you don't need gradients when you're just making predictions, not training. This saves memory and speeds things up.

**Math Behind It:**
This is basic calculus. If `y = x² + 2x`, then `dy/dx = 2x + 2` (the power rule: derivative of `x²` is `2x`; derivative of `2x` is `2`). At `x = 3`: `2(3) + 2 = 8`. Autograd doesn't "look up" this formula — it computes it step by step via the chain rule as it walks back through the recorded operations (`x**2`, then `+ 2*x`).

## 7. `nn.Module`

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(32, 10)

output = model(x)
print(output.shape)
```

**Expected Output:**

```text
torch.Size([32, 1])
```

Use `model(x)`, not `model.forward(x)`, because PyTorch's module machinery handles hooks and other behavior around the call.

**Key Terms:**
- **`nn.Module`** — the base class for anything with learnable weights. Subclassing it and defining `__init__` (create the layers) and `forward` (define how data flows through them) is the standard pattern for every model in this guide.
- **`__init__` vs `forward`** — `__init__` builds the layers once; `forward` describes the computation each time you call the model with input data.
- **Batch dimension** — the first dimension (`32` here) represents 32 independent examples processed together. Almost every tensor in PyTorch has a batch dimension first.
- **`model(x)` vs `model.forward(x)`** — calling `model(x)` actually calls Python's `__call__`, which runs `forward` *plus* extra bookkeeping (like triggering hooks). Calling `.forward()` directly skips that bookkeeping, which can silently break things.

## 8. Parameters

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

**Expected Output:**

```text
linear.weight torch.Size([1, 10])
linear.bias torch.Size([1])
```

Trainable parameter count:

```python
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

**Expected Output:** `trainable` evaluates to `11` (10 weights + 1 bias for the `Linear(10, 1)` layer above).

**Key Terms:**
- **Parameter** — a learnable tensor (weight or bias) that the optimizer updates during training.
- **`named_parameters()`** — iterates over every parameter along with a string name showing where it lives in the model (useful for debugging which layers are frozen vs. trainable).
- **`numel()`** — "number of elements": the total count of values in a tensor, i.e. the product of its shape (`1 × 10 = 10` for the weight above).

## 9. Linear Layers

```python
layer = nn.Linear(
    in_features=10,
    out_features=5,
)

x = torch.randn(32, 10)
y = layer(x)

print(y.shape)  # [32, 5]
```

**Expected Output:**

```text
torch.Size([32, 5])
```

Conceptually:

`y = xWᵀ + b`

**Key Terms:**
- **`nn.Linear`** — also called a "fully connected" or "dense" layer. It maps every input feature to every output feature via a learned weight matrix.
- **`in_features` / `out_features`** — the size of the input vector and output vector per example. Internally, the weight matrix `W` has shape `[out_features, in_features]`.
- **Bias** — a learned constant added after the matrix multiply, giving the layer one extra degree of freedom (like the `+b` in `y = mx + b`).

**Math Behind It:**
For a single input row `x` of size 10, `Wᵀ` (W transposed) has shape `[10, 5]`, so `x @ Wᵀ` produces a size-5 output, then `b` (size 5) is added element-wise. Applied to a batch of 32 rows at once, the shapes work out to `[32, 10] @ [10, 5] + [5] = [32, 5]`.

## 10. Activations

```python
relu = nn.ReLU()
gelu = nn.GELU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
```

GELU is commonly used in Transformer architectures.

**Key Terms:**
- **Activation function** — a nonlinear function applied after a linear layer. Without it, stacking linear layers would collapse into one big linear layer — activations are what let a network learn complex, non-linear patterns.
- **ReLU** — `max(0, x)`. Outputs `x` if positive, else `0`. Simple and fast, but can "die" (get stuck outputting 0) for negative inputs.
- **GELU** — a smoother version of ReLU that also lets small negative values through, roughly `x × Φ(x)` where `Φ` is the standard normal cumulative distribution function. Used in BERT, GPT, and most modern Transformers.
- **Sigmoid** — squashes any real number into `(0, 1)`, formula `1 / (1 + e^-x)`. Used for binary probabilities.
- **Tanh** — squashes any real number into `(-1, 1)`, formula `(e^x - e^-x) / (e^x + e^-x)`.

## 11. Sequential

```python
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

**Key Terms:**
- **`nn.Sequential`** — a container that chains layers in order: the output of each layer feeds directly into the next. It's a shortcut for writing a full `nn.Module` class when your model is just "one thing after another" with no branching.

## 12. Loss Functions

Regression:

```python
loss_fn = nn.MSELoss()
```

Multiclass classification:

```python
loss_fn = nn.CrossEntropyLoss()
```

Binary classification:

```python
loss_fn = nn.BCEWithLogitsLoss()
```

Important: `CrossEntropyLoss` expects raw logits, so normally do not apply softmax before passing logits to it.

**Key Terms:**
- **Loss function** — measures the gap between the model's prediction and the true label as a single number that training tries to minimize.
- **Logits** — the raw, un-normalized output scores of a model before any activation like softmax or sigmoid is applied. `CrossEntropyLoss` applies the softmax internally for numerical stability, so feeding it already-softmaxed values would double-apply it and break training.
- **MSE (Mean Squared Error)** — average of `(prediction - target)²` over all examples. Used for regression (predicting continuous numbers).
- **Cross-Entropy** — measures the difference between the predicted probability distribution and the true class. Used for multi-class classification.
- **BCE (Binary Cross-Entropy) with Logits** — like cross-entropy but for a single yes/no probability; the "WithLogits" version applies sigmoid internally for the same numerical-stability reason.

**Math Behind It:**
- MSE: `loss = (1/N) Σ (ŷᵢ - yᵢ)²` — squaring penalizes large errors more than small ones.
- Cross-Entropy: `loss = -log(p_correct_class)` — the loss is small when the model assigns high probability to the correct class, and grows sharply (toward infinity) as that probability approaches 0.

## 13. Optimizers

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)
```

AdamW is particularly common in Transformer training.

**Key Terms:**
- **Optimizer** — the algorithm that updates each parameter using its gradient, once per training step.
- **Learning rate (`lr`)** — how big a step the optimizer takes each update. Too high and training diverges; too low and training is painfully slow.
- **Adam** — an optimizer that keeps a running average of past gradients (momentum) and adapts the effective learning rate per parameter, making it more robust than plain gradient descent.
- **AdamW** — Adam with "decoupled weight decay": weight decay (a regularization technique that shrinks weights slightly each step to prevent overfitting) is applied separately from the gradient update rather than mixed into it, which tends to work better in practice.

**Math Behind It:**
Plain gradient descent is `w = w - lr × gradient`. Adam refines this by tracking a moving average of the gradient (`m`, "first moment") and of the squared gradient (`v`, "second moment"), then updates roughly as `w = w - lr × m / (√v + ε)` — dividing by `√v` means parameters with consistently large gradients get smaller effective steps, and vice versa.

## 14. Standard Training Loop

```python
model.train()

for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    predictions = model(x)
    loss = loss_fn(predictions, y)

    loss.backward()
    optimizer.step()
```

A common memory-friendly variant is:

```python
optimizer.zero_grad(set_to_none=True)
```

**Key Terms:**
- **`model.train()`** — puts the model in training mode (affects layers like Dropout/BatchNorm — see Section 15).
- **`optimizer.zero_grad()`** — clears old gradients before computing new ones. Gradients **accumulate** by default in PyTorch (they add up across `.backward()` calls), so skipping this would mix gradients from the previous batch into the current one.
- **`loss.backward()`** — computes gradients of the loss with respect to every parameter (via autograd).
- **`optimizer.step()`** — actually updates the parameters using those gradients.
- **`set_to_none=True`** — instead of resetting gradients to `0`, it sets them to `None`, which slightly reduces memory usage and speeds things up (PyTorch handles the `None` case internally on the next backward pass).

**Math Behind It:**
This loop is one iteration of **stochastic gradient descent**: "stochastic" because each step only sees one mini-batch (a small random subset of data) rather than the entire dataset, which makes training much faster while still converging toward good parameters on average.

## 15. Validation / Inference

```python
model.eval()

total_loss = 0.0

with torch.inference_mode():
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)
        loss = loss_fn(predictions, y)

        total_loss += loss.item()
```

Know why `model.train()` and `model.eval()` matter: layers such as dropout and batch normalization behave differently between training and evaluation.

**Key Terms:**
- **`model.eval()`** — switches the model to evaluation mode. It does *not* disable gradient tracking by itself — that's what `inference_mode()`/`no_grad()` are for.
- **Dropout** — during training, randomly zeroes out a fraction of neurons each forward pass (regularization, prevents overfitting). During eval, it's turned off and all neurons are used.
- **Batch Normalization** — during training, normalizes each batch using that batch's own mean/variance while also tracking a running average; during eval, it uses the stored running average instead, since a validation batch might not be statistically representative.
- **`.item()`** — pulls a single-element tensor out as a plain Python number, useful for accumulating/logging without keeping it attached to the autograd graph.

## 16. Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)
```

Production-oriented options include `num_workers` and `pin_memory`.

**Key Terms:**
- **`Dataset`** — a class that knows how to fetch one example at a time. You must implement `__len__` (total number of examples) and `__getitem__` (return example `i`).
- **`DataLoader`** — wraps a `Dataset` and automatically batches, shuffles, and (optionally) parallelizes loading examples into mini-batches for training.
- **`batch_size`** — how many examples are grouped into one tensor per training step.
- **`shuffle=True`** — randomizes example order each epoch, which prevents the model from learning spurious patterns tied to data ordering.
- **`num_workers`** — number of background CPU processes used to load data in parallel, so the GPU isn't left waiting on data loading.
- **`pin_memory`** — allocates loaded batches in "page-locked" memory, which speeds up the CPU→GPU transfer.

## 17. Saving and Loading

```python
torch.save(model.state_dict(), "model.pt")
```

Load:

```python
model = SimpleModel()

state = torch.load(
    "model.pt",
    map_location=device,
)

model.load_state_dict(state)
```

Prefer `state_dict()` for normal model checkpoints.

**Key Terms:**
- **`state_dict()`** — a Python dictionary mapping each layer's name to its current parameter tensor (weights/biases). It holds *only* the numbers, not the model's code/architecture.
- **Why prefer `state_dict` over saving the whole model** — saving the whole model (`torch.save(model, ...)`) pickles the class definition too, which is fragile across code changes and Python/PyTorch versions. Saving just the `state_dict` is more portable: you re-create the model class in code, then load the weights into it.
- **`map_location`** — tells `torch.load` which device to put the loaded tensors on, useful when saving on a GPU machine and loading on a CPU-only machine (or vice versa).

# CNN

## 18. CNN Mental Model

CNNs are good at learning local spatial patterns.

```text
Input
  ↓
Convolution
  ↓
Activation
  ↓
Pooling
  ↓
Convolution
  ↓
Activation
  ↓
Pooling
  ↓
Classifier
```

PyTorch image layout is normally:

`[batch, channels, height, width]`

**Key Terms:**
- **CNN (Convolutional Neural Network)** — a network architecture built around convolution layers, which are especially good at images because they exploit **spatial locality** (nearby pixels are related) and **weight sharing** (the same small filter is reused across the whole image).
- **Channels** — for a color image, typically 3 (Red, Green, Blue). Feature maps produced by convolution layers have many more channels (e.g. 32, 64) representing different learned patterns.
- **Pooling** — downsamples a feature map (e.g. `MaxPool2d` takes the max value in each small region), reducing spatial size and computation while keeping the strongest signals.

**Math Behind It:**
A convolution slides a small learnable filter (kernel) across the image and, at each position, computes a weighted sum of the pixels it covers (a dot product), producing one output value per position. This is what lets the same filter detect the same pattern (e.g. an edge) no matter where in the image it appears.

## 19. Conv2d

```python
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    padding=1,
)

x = torch.randn(32, 3, 64, 64)
y = conv(x)

print(y.shape)
```

**Expected Output:**

```text
torch.Size([32, 16, 64, 64])
```

With `padding=1` and stride 1, the spatial size remains 64×64.

**Key Terms:**
- **`in_channels` / `out_channels`** — number of channels going in (3, for RGB) and the number of learned filters/feature maps going out (16 here).
- **`kernel_size`** — the height/width of the sliding filter, here `3×3`.
- **`padding`** — how many pixels of zero-padding to add around the image border, so the filter can be centered on edge pixels too, and so the output size can be controlled.
- **Stride** (not set here, defaults to 1) — how many pixels the filter moves at each step. Stride 2 would roughly halve the output size.

**Math Behind It:**
Output spatial size follows: `output = (input − kernel_size + 2×padding) / stride + 1`. Here: `(64 − 3 + 2×1) / 1 + 1 = 64`, which is why the spatial dimensions (64×64) stay unchanged — only the channel count changes (3 → 16).

## 20. CNN Classifier

```python
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

For 32×32 input:

```text
32×32 → pool → 16×16 → pool → 8×8
```

**Expected Output:** for input `x = torch.randn(32, 3, 32, 32)` (batch of 32 RGB 32×32 images), `model(x).shape` is `torch.Size([32, 10])` — 32 examples, 10 class scores each.

**Key Terms:**
- **`MaxPool2d(2)`** — takes the maximum value in every non-overlapping 2×2 block, halving both height and width.
- **`nn.Flatten()`** — collapses all dimensions except the batch dimension into one long vector, so a `[batch, 64, 8, 8]` feature map becomes `[batch, 64×8×8] = [batch, 4096]` before entering the `Linear` layer.
- **Why `64 * 8 * 8`** — after two `MaxPool2d(2)` layers, a 32×32 image shrinks to 8×8 (32→16→8), and the last conv layer outputs 64 channels, so the flattened size must exactly match `64 × 8 × 8 = 4096`.

## 21. Better CNN with Adaptive Pooling

Adaptive pooling avoids hardcoding image dimensions.

```python
class BetterCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

**Expected Output:** whatever the input's height/width, `AdaptiveAvgPool2d((1, 1))` always produces a `[batch, 64, 1, 1]` feature map — e.g. for `x = torch.randn(32, 3, 96, 96)`, `model(x).shape` is `torch.Size([32, 10])`, same as with a 32×32 input.

**Key Terms:**
- **`AdaptiveAvgPool2d((1, 1))`** — instead of a fixed pooling window, it automatically figures out the window size needed to produce exactly a `1×1` output, averaging the entire feature map into one value per channel. This is why `BetterCNN` doesn't need to hardcode `8 * 8` like the previous model — it works for any input image size.

# RNN / LSTM

## 22. RNN Mental Model

An RNN processes a sequence recurrently:

`h_t = f(x_t, h_(t-1))`

With `batch_first=True`, sequence tensors normally use:

`[batch, sequence_length, feature_dimension]`

**Key Terms:**
- **RNN (Recurrent Neural Network)** — a network that processes a sequence one step at a time, carrying forward a **hidden state** that acts as a summary/memory of everything seen so far.
- **Hidden state (`h_t`)** — the network's running memory at time step `t`, updated based on the current input `x_t` and the previous hidden state `h_(t-1)`.
- **`batch_first=True`** — a constructor argument that puts the batch dimension first in the expected tensor shape (`[batch, seq, feature]`) instead of PyTorch's historical default (`[seq, batch, feature]`), which is usually more intuitive.

**Math Behind It:**
The recurrence `h_t = f(x_t, h_(t-1))` typically expands to `h_t = tanh(W_x·x_t + W_h·h_(t-1) + b)` — the new hidden state is a nonlinear combination of the current input and the previous hidden state, using two learned weight matrices shared across *every* time step.

## 23. RNN Classifier

```python
class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(self, tokens):
        x = self.embedding(tokens)

        output, hidden = self.rnn(x)

        last_hidden = hidden[-1]

        return self.classifier(last_hidden)
```

**Expected Output:** for `tokens = torch.randint(0, vocab_size, (16, 20))` (batch of 16 sequences, each 20 tokens long), the shapes through the model are:

```text
x (after embedding):        [16, 20, embedding_dim]
output (all hidden states): [16, 20, hidden_dim]
hidden (final hidden state): [1, 16, hidden_dim]
last_hidden = hidden[-1]:   [16, hidden_dim]
final output:                [16, num_classes]
```

**Key Terms:**
- **`nn.Embedding`** — a lookup table mapping each integer token ID to a learned dense vector, converting discrete words/tokens into continuous representations a network can process.
- **`output` vs `hidden`** — `output` contains the hidden state at *every* time step; `hidden` contains only the *final* hidden state (per layer). Since this model classifies the whole sequence, it only needs the final summary, `hidden[-1]`.
- **`hidden[-1]`** — indexes the last layer's final hidden state (here there's only 1 layer, so `hidden` has shape `[1, batch, hidden_dim]` and `hidden[-1]` gives `[batch, hidden_dim]`).

## 24. LSTM

LSTM adds gating mechanisms to better preserve information across long sequences.

```python
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(self, tokens):
        x = self.embedding(tokens)

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        return self.classifier(last_hidden)
```

**Expected Output:** structurally identical to the RNN classifier above, except `self.lstm(x)` returns an extra `cell` tensor alongside `hidden`, also of shape `[1, batch, hidden_dim]`.

Bidirectional LSTM:

```python
self.lstm = nn.LSTM(
    embedding_dim,
    hidden_dim,
    batch_first=True,
    bidirectional=True,
)

self.classifier = nn.Linear(
    hidden_dim * 2,
    num_classes,
)
```

**Key Terms:**
- **LSTM (Long Short-Term Memory)** — an RNN variant with a separate **cell state** and three learned **gates** (forget, input, output) that control what information to keep, add, or output at each step. This solves the "vanishing gradient" problem where plain RNNs struggle to remember information from many steps ago.
- **Cell state (`cell`)** — a second memory channel that runs through the sequence with only minor, gate-controlled modifications, acting as a long-term memory "conveyor belt" (separate from the hidden state, which is more like short-term/output memory).
- **Gates** — small neural networks (a `Linear` layer + sigmoid) that output values between 0 and 1, acting as "how much to let through" switches. The **forget gate** decides what to discard from the cell state, the **input gate** decides what new information to add, and the **output gate** decides what to expose as the hidden state.
- **Bidirectional** — runs two LSTMs over the sequence, one left-to-right and one right-to-left, then concatenates their outputs — giving each position context from both past *and* future tokens. This is why `hidden_dim * 2` is needed for the classifier: the hidden states from both directions are stacked together.

**Math Behind It (gates, simplified):**
`forget_gate = sigmoid(W_f · [h_(t-1), x_t])`, and similarly for the input and output gates. Because sigmoid outputs are between 0 and 1, multiplying the cell state by the forget gate acts like a dimmer switch — values near 0 erase that part of memory, values near 1 preserve it — which is what lets LSTMs retain information over long sequences far better than plain RNNs.

> **Interview note:** with `bidirectional=True`, `hidden` has shape `[num_layers × 2, batch, hidden_dim]` (forward and backward stacked). Taking just `hidden[-1]` only grabs the *backward* direction's final state — to use both directions you'd concatenate `hidden[-2]` (forward) and `hidden[-1]` (backward) before the classifier.

# Transformer

## 25. Why Transformers

RNNs process tokens sequentially. Transformers use attention so tokens can directly interact and training can be highly parallelized.

**Key Terms:**
- **Sequential bottleneck** — an RNN must finish processing token `t-1` before it can process token `t`, since each step depends on the previous hidden state. This makes RNNs slow to train on long sequences, since GPUs can't parallelize across time steps.
- **Attention** — a mechanism that lets every token look directly at every other token in one shot, regardless of distance, and lets all tokens be processed simultaneously on a GPU — this is the key reason Transformers train much faster than RNNs on large datasets.

## 26. Embeddings

```python
embedding = nn.Embedding(
    num_embeddings=10_000,
    embedding_dim=256,
)

tokens = torch.tensor([
    [10, 20, 30],
    [40, 50, 60],
])

x = embedding(tokens)

print(x.shape)  # [2, 3, 256]
```

**Expected Output:**

```text
torch.Size([2, 3, 256])
```

Typical NLP representation:

`[batch, sequence, embedding_dimension]`

**Key Terms:**
- **`num_embeddings`** — the vocabulary size: how many distinct tokens the lookup table needs an entry for (10,000 here).
- **`embedding_dim`** — the length of the dense vector representing each token (256 here). Each of the 10,000 tokens gets its own learned 256-length vector.
- Here, `tokens` is `[2, 3]` (2 sequences of 3 tokens); the embedding layer replaces every integer token ID with its 256-dim vector, producing `[2, 3, 256]`.

## 27. Scaled Dot-Product Attention

The core equations are:

`Q = XW_Q`

`K = XW_K`

`V = XW_V`

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k))V`

Implementation:

```python
import math

def scaled_dot_product_attention(
    query,
    key,
    value,
    mask=None,
):
    d_k = query.size(-1)

    scores = (
        query @ key.transpose(-2, -1)
    ) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

    weights = torch.softmax(
        scores,
        dim=-1,
    )

    return weights @ value
```

**Expected Output:** for `query`, `key`, `value` each of shape `[batch, seq_len, d_k]`, e.g. `[2, 4, 8]`, the function returns a tensor of shape `[2, 4, 8]` — same shape as `query`/`value`. `weights` (before the final `@ value`) has shape `[2, 4, 4]`: one attention weight per (query position, key position) pair.

Why divide by `sqrt(d_k)`?

Without scaling, dot products tend to grow with dimensionality, making softmax too sharp and gradients less stable.

**Key Terms:**
- **Query, Key, Value (Q, K, V)** — three different learned projections of the same input `X`. Intuitively: the **query** represents "what am I looking for," the **key** represents "what do I contain" (used to match against queries), and the **value** is "what information do I actually pass along" once a match is found.
- **Attention scores** — the raw similarity between each query and each key, computed via dot product (`Q @ Kᵀ`). A high score means that key is very relevant to that query.
- **`softmax`** — converts a row of raw scores into a probability distribution (all values between 0 and 1, summing to 1), so attention weights represent "how much focus" to put on each other token.
- **`masked_fill`** — replaces scores at masked-out positions with `-∞` *before* softmax, so that after softmax those positions get essentially 0 weight (e.g. used for padding tokens or future tokens in causal attention — see Section 33).

**Math Behind It:**
`d_k` is the dimensionality of each query/key vector. As `d_k` grows, dot products between random vectors tend to have larger variance (their magnitude scales roughly with `√d_k`), which pushes softmax inputs to extreme values, making the resulting probability distribution overly "peaked" (nearly one-hot) and causing very small gradients during backpropagation. Dividing by `√d_k` counteracts this growth, keeping the scores — and therefore the softmax output and gradients — well-behaved regardless of dimension size.

## 28. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                float("-inf"),
            )

        weights = torch.softmax(scores, dim=-1)

        context = weights @ v

        context = context.transpose(1, 2).contiguous()

        batch_size, seq_len, _ = x.shape

        context = context.view(
            batch_size,
            seq_len,
            self.d_model,
        )

        return self.out_proj(context)
```

Important shapes:

```text
Input:             [B, T, D]
After head split:  [B, H, T, Dh]
Attention scores:  [B, H, T, T]
Context:           [B, H, T, Dh]
Merged:            [B, T, D]
```

**Expected Output:** for `x = torch.randn(4, 10, 256)` with `num_heads=8` (`B=4, T=10, D=256, H=8, Dh=32`), `model(x).shape` is `torch.Size([4, 10, 256])` — the output shape always matches the input shape, which is what lets Transformer blocks be stacked.

**Key Terms:**
- **Multi-head attention** — instead of computing one attention pattern, it computes `num_heads` smaller attention patterns in parallel (each on a `head_dim`-sized slice of the embedding), then concatenates the results. Each head can learn to specialize in a different kind of relationship (e.g. one head might track syntax, another might track coreference).
- **`head_dim`** — `d_model // num_heads`: since the total representation size (`d_model`) is split evenly across heads, more heads means each head works with a smaller slice.
- **`split_heads`** — reshapes `[B, T, D]` into `[B, H, T, Dh]` and transposes so attention can be computed independently per head (`T` and `Dh` become the "active" trailing dimensions for the `@` matmul).
- **`.contiguous()`** — after a `.transpose()`, a tensor's memory layout is no longer contiguous, so a subsequent `.view()` would fail; `.contiguous()` makes a properly laid-out copy first.

## 29. Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)
```

**Expected Output:** for `x` of shape `[B, T, d_model]`, `model(x).shape` is also `[B, T, d_model]` — the network expands to `hidden_dim` (typically `4 × d_model`) internally, then projects back down.

**Key Terms:**
- **Feed-Forward Network (FFN)** — a small two-layer MLP applied independently to *each* token position (attention is what lets tokens interact; the FFN then processes each token's resulting representation on its own). This is where a large fraction of a Transformer's total parameters live.
- **`hidden_dim`** — the FFN's intermediate width, commonly 4× `d_model`, giving the network more capacity to transform each token's representation before compressing back down.

## 30. Transformer Block

A pre-norm Transformer block can be implemented as:

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model,
            num_heads,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(
            d_model,
            ff_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(
            self.norm1(x),
            mask,
        )

        x = x + self.dropout(attention_output)

        ff_output = self.ff(self.norm2(x))

        x = x + self.dropout(ff_output)

        return x
```

Architecture:

```text
x
 ↓
LayerNorm
 ↓
Self-Attention
 ↓
Residual Add
 ↓
LayerNorm
 ↓
FFN
 ↓
Residual Add
```

**Expected Output:** for `x = torch.randn(4, 10, 256)` with matching `d_model=256`, `block(x).shape` is `torch.Size([4, 10, 256])` — again, shape-preserving, so blocks can be stacked arbitrarily deep (see Section 32).

**Key Terms:**
- **`LayerNorm`** — normalizes each token's feature vector to have mean 0 and variance 1 (then applies a learned scale and shift), which stabilizes training in deep networks. Unlike BatchNorm, it normalizes *across features* for a single example, not across the batch — so its behavior doesn't depend on batch size.
- **Residual connection (`x + ...`)** — adds the block's input back to its output ("skip connection"). This gives gradients a direct path backward through the network, making very deep Transformers trainable.
- **Pre-norm** — applying `LayerNorm` *before* attention/FFN (as here) rather than after. Pre-norm tends to give more stable training for deep Transformers, which is why most modern LLMs use it.
- **`nn.Dropout`** — randomly zeroes some values during training (regularization) to prevent the model from over-relying on any specific pathway.

**Math Behind It:**
`LayerNorm(x) = γ × (x − mean(x)) / √(var(x) + ε) + β`, computed per-token across its feature dimension, where `γ` and `β` are learned parameters letting the network undo the normalization if that's actually better for a given layer.

## 31. Token + Position Embeddings

Attention itself does not encode sequence order. A simple learned position embedding is:

```python
class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model,
        )

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape

        positions = torch.arange(
            seq_len,
            device=tokens.device,
        )

        return (
            self.token_embedding(tokens)
            + self.position_embedding(positions)
        )
```

Modern LLMs may use alternatives such as RoPE.

**Expected Output:** for `tokens` of shape `[batch, seq_len]`, the output shape is `[batch, seq_len, d_model]` — `position_embedding(positions)` produces `[seq_len, d_model]`, which broadcasts (Section 4) across the batch dimension when added to the token embeddings.

**Key Terms:**
- **Positional embedding** — since attention treats all tokens symmetrically (it has no built-in notion of "first" or "second"), a separate signal encoding *where* each token sits in the sequence must be added in, otherwise "the cat sat" and "sat the cat" would look identical to the attention mechanism.
- **Learned position embedding** — here, position `0, 1, 2, ...` each get their own trainable vector, just like a token would, looked up from an embedding table.
- **RoPE (Rotary Position Embedding)** — a more modern alternative (used in LLaMA, GPT-NeoX, etc.) that encodes position by *rotating* the query/key vectors by an angle proportional to their position, which has the nice property of naturally encoding *relative* distance between tokens rather than only absolute position.

## 32. Transformer Classifier

```python
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=4,
        num_classes=5,
    ):
        super().__init__()

        self.embedding = TokenPositionEmbedding(
            vocab_size,
            max_seq_len,
            d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                ff_dim,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(self, tokens):
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        pooled = x.mean(dim=1)

        return self.classifier(pooled)
```

**Expected Output:** for `tokens = torch.randint(0, vocab_size, (8, 20))` (batch of 8 sequences, 20 tokens each), the shapes are:

```text
x (after embedding):   [8, 20, 256]
x (after 4 blocks):    [8, 20, 256]   # unchanged, blocks are shape-preserving
pooled (mean over dim=1): [8, 256]
final output:          [8, 5]
```

**Key Terms:**
- **Mean pooling (`x.mean(dim=1)`)** — averages the representations of all tokens in the sequence into a single vector per example, used here to get one fixed-size vector for classification regardless of sequence length. (Alternatives include using a special `[CLS]` token's representation instead, as BERT does.)
- **Stacking blocks (`nn.ModuleList`)** — because each `TransformerBlock` preserves shape (`[B, T, D]` in, `[B, T, D]` out), any number of them can be chained — this depth is what lets Transformers build increasingly abstract representations layer by layer.

## 33. Causal Mask

Decoder-only LLMs such as GPT-style models cannot attend to future tokens.

```python
def causal_mask(seq_len, device):
    return torch.tril(
        torch.ones(
            seq_len,
            seq_len,
            device=device,
        )
    )
```

For four tokens:

```text
A
A B
A B C
A B C D
```

The upper triangle is masked.

**Expected Output:** `causal_mask(4, "cpu")` returns:

```text
tensor([[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]])
```

**Key Terms:**
- **Causal (autoregressive) masking** — ensures token `t` can only attend to tokens `≤ t`, never future ones. This matches how text is generated: one token at a time, using only what's been produced so far.
- **`torch.tril`** — "triangular lower": zeroes out everything above the main diagonal, keeping only the lower-triangular part (including the diagonal). Row `i` having `1`s in columns `0..i` exactly encodes "token `i` can see tokens `0` through `i`."
- Combined with `masked_fill(mask == 0, float("-inf"))` from Section 27, the `0` entries become `-∞` before softmax, so after softmax those future positions get essentially 0 attention weight.

## 34. Built-in Transformer

You should understand both implementation and the high-level API:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    batch_first=True,
)

encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6,
)

x = torch.randn(32, 100, 256)
output = encoder(x)
```

**Expected Output:** `output.shape` is `torch.Size([32, 100, 256])` — same shape-preserving pattern as the hand-built `TransformerBlock`.

**Key Terms:**
- **`nn.TransformerEncoderLayer`** — PyTorch's built-in equivalent of the `TransformerBlock` from Section 30 (multi-head attention + FFN + residuals + norm), production-tested and optimized.
- **`nn.TransformerEncoder`** — stacks `num_layers` copies of the encoder layer, equivalent to the `nn.ModuleList` loop in Section 32.
- **`nhead`** — PyTorch's built-in naming for what this guide calls `num_heads`.

# LoRA

## 35. LoRA Concept

Full fine-tuning updates the pretrained weight matrix:

`W' = W + ΔW`

LoRA freezes `W` and learns a low-rank update:

`ΔW = BA`

So:

`y = Wx + BAx`

With scaling:

`y = Wx + (alpha/r)BAx`

where `r` is a small rank.

```text
Pretrained W ───────── frozen
       +
A → B ──────────────── trainable
```

**Key Terms:**
- **LoRA (Low-Rank Adaptation)** — a parameter-efficient fine-tuning technique: instead of updating the full pretrained weight matrix `W` (which may have millions of parameters), you freeze `W` entirely and learn a much smaller *low-rank* correction `ΔW = BA` on top of it.
- **Rank (`r`)** — the "bottleneck" dimension of the low-rank matrices `A` and `B`. A small `r` (e.g. 8) means `A` and `B` together have far fewer parameters than `W`, while still being able to represent a useful update direction.
- **`alpha`** — a scaling hyperparameter controlling how strongly the LoRA update `BA` influences the output relative to the frozen base weight.

**Math Behind It:**
If `W` has shape `[out, in]`, a **low-rank matrix** `BA` (with `B` shaped `[out, r]` and `A` shaped `[r, in]`) can only represent updates that live in an `r`-dimensional subspace instead of the full `out × in` space — but empirically, the useful adaptation needed to fine-tune a model to a new task tends to fit well within a small subspace like this, which is why LoRA works nearly as well as full fine-tuning at a fraction of the trainable parameters. This mirrors the linear-algebra fact that any `[out, in]` matrix can be decomposed as a product of a `[out, r]` and `[r, in]` matrix if you allow `r` to be as large as `min(out, in)`; LoRA deliberately picks a much smaller `r`.

## 36. LoRA Linear Layer

```python
class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank=8,
        alpha=16,
    ):
        super().__init__()

        self.base = base_layer

        for param in self.base.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_a = nn.Parameter(
            torch.randn(
                rank,
                in_features,
            ) * 0.01
        )

        self.lora_b = nn.Parameter(
            torch.zeros(
                out_features,
                rank,
            )
        )

        self.scale = alpha / rank

    def forward(self, x):
        base_output = self.base(x)

        lora_output = (
            x
            @ self.lora_a.T
            @ self.lora_b.T
        )

        return (
            base_output
            + self.scale * lora_output
        )
```

**Expected Output:** wrapping `nn.Linear(4096, 4096)` with `rank=8`, for `x` of shape `[batch, 4096]`:

```text
base_output:  [batch, 4096]
x @ lora_a.T: [batch, 4096] @ [4096, 8]  -> [batch, 8]
   @ lora_b.T:      [batch, 8] @ [8, 4096] -> [batch, 4096]
final output: [batch, 4096]
```

**Key Terms:**
- **`param.requires_grad = False`** — this is how `W` (the base layer) gets "frozen": autograd will still compute a forward pass through it, but no gradient is stored for its parameters, so the optimizer never updates them.
- **`lora_a`, `lora_b`** — the trainable low-rank matrices `A` and `B`. `lora_a` is initialized with small random values, `lora_b` is initialized to all zeros (explained in Section 37).
- **`self.scale = alpha / rank`** — the `alpha/r` scaling factor from the formula in Section 35, applied to the LoRA correction before adding it to the frozen base output.

## 37. Why Initialize B to Zero?

Initially:

`BA = 0`

Therefore:

`W' = W`

The LoRA-augmented model initially behaves like the pretrained model and learns the adaptation during training.

**Key Terms:**
- **Zero-init for `B`** — since `lora_b` starts as all zeros, `BA = B @ A = 0` regardless of what `A` contains, so at the very start of training the LoRA correction contributes exactly nothing.

**Math Behind It:**
This matters because it guarantees training starts from the *exact* pretrained model's behavior (`W' = W + 0 = W`), rather than from some random perturbation of it. If both `A` and `B` were randomly initialized, the model's outputs at step 0 would already be scrambled by a random `BA` term, potentially hurting early training. Gradients still flow into both `A` and `B` from the very first backward pass (because `∂loss/∂B` depends on `A`, which is non-zero), so training proceeds normally from that clean starting point.

## 38. Parameter Savings

For a 4096×4096 linear layer:

Full weight:

`4096 × 4096 ≈ 16.8 million parameters`

Rank-8 LoRA:

`4096 × 8 + 8 × 4096 = 65,536 parameters`

This dramatically reduces trainable parameters, gradient memory, optimizer memory, and adapter checkpoint size.

**Math Behind It:**
`4096 × 4096 = 16,777,216` parameters for the full weight matrix. The rank-8 LoRA matrices contribute `4096 × 8 = 32,768` (matrix `A`) plus `8 × 4096 = 32,768` (matrix `B`), totaling `65,536` — about **0.39%** of the full matrix's parameter count (`65,536 / 16,777,216 ≈ 0.0039`). Since Adam-style optimizers store roughly 2 extra numbers per trainable parameter (momentum + variance), this also cuts optimizer memory by roughly the same ~256× factor for this layer.

## 39. Applying LoRA to Attention

Given:

```python
self.q_proj = nn.Linear(d_model, d_model)
self.k_proj = nn.Linear(d_model, d_model)
self.v_proj = nn.Linear(d_model, d_model)
self.out_proj = nn.Linear(d_model, d_model)
```

You can replace selected projections:

```python
self.q_proj = LoRALinear(
    self.q_proj,
    rank=8,
)

self.v_proj = LoRALinear(
    self.v_proj,
    rank=8,
)
```

Depending on the model/task, adapters can also target K, O, and MLP projections.

**Key Terms:**
- **Target modules** — which of the model's linear layers get wrapped with `LoRALinear`. Q and V projections are the most common default (from the original LoRA paper), since adapting how queries and values are computed tends to give strong results, but K, output-projection, and MLP layers can also be targeted for more adaptation capacity.

## 40. Verify Trainable Parameters

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

**Expected Output:** only the LoRA parameters print, e.g.:

```text
q_proj.lora_a
q_proj.lora_b
v_proj.lora_a
v_proj.lora_b
```

(Everything else — the frozen base weights — is silently skipped, since their `requires_grad` is `False`.)

Count:

```python
trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Trainable: {trainable:,}")
print(f"Total: {total:,}")
print(f"Percent: {100 * trainable / total:.4f}%")
```

**Expected Output:** (exact numbers depend on your model size — illustrative example for a small model)

```text
Trainable: 65,536
Total: 16,842,752
Percent: 0.3891%
```

**Key Terms:**
- **`f"{trainable:,}"`** — Python f-string formatting: the `,` adds thousands separators (e.g. `65536` → `65,536`) purely for readability.
- This percentage is the number interviewers often ask for directly — it's the concrete, quantifiable payoff of using LoRA over full fine-tuning.

## 41. LoRA Training Loop

```python
optimizer = torch.optim.AdamW(
    (
        p
        for p in model.parameters()
        if p.requires_grad
    ),
    lr=1e-4,
)

model.train()

for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()

    logits = model(x)

    loss = loss_fn(logits, y)

    loss.backward()
    optimizer.step()
```

Only the LoRA parameters are updated.

**Key Terms:**
- **Filtering `model.parameters()`** — passing only `requires_grad=True` parameters to the optimizer means it only tracks momentum/variance state (see Section 13's math) for the small LoRA matrices, not the frozen base weights — this is where the optimizer-memory savings from Section 38 actually materialize.
- Note the lower learning rate (`1e-4` vs. the `1e-3` used for full training in Section 13) — LoRA fine-tuning typically uses a smaller or comparable learning rate since it's making a targeted adjustment to an already-good pretrained model, not learning from scratch.

## 42. Merge LoRA

The merged weight is:

`W' = W + (alpha/r)BA`

Example:

```python
def merged_weight(layer):
    delta = (
        layer.lora_b
        @ layer.lora_a
    ) * layer.scale

    return layer.base.weight + delta
```

Merging can remove the extra adapter computation during inference.

**Expected Output:** `merged_weight(layer)` returns a single tensor of the same shape as `layer.base.weight` (e.g. `[4096, 4096]`), combining the frozen base weight with the scaled LoRA correction into one ordinary weight matrix.

**Key Terms:**
- **Merging** — once training is done, you can precompute `B @ A × scale` and add it directly into `W`, producing a single ordinary `nn.Linear` weight. This removes the extra matrix multiplications LoRA introduces at inference time, so the merged model runs exactly as fast as the original, unmodified model — at the cost of losing the ability to easily swap adapters (since the base and adapter weights are now fused together).

## 43. QLoRA

LoRA:

```text
FP16/BF16 frozen base model
        +
trainable LoRA adapters
```

QLoRA:

```text
quantized frozen base model
        +
trainable LoRA adapters
```

The important idea is that the base model is heavily quantized to reduce memory while LoRA adapters remain trainable.

**Key Terms:**
- **QLoRA** — combines LoRA with **quantization**: the frozen base model's weights are stored in a much lower-precision format (commonly 4-bit, via the NF4 data type) instead of 16-bit floats, drastically cutting the memory needed to even load the base model, while the small trainable LoRA adapters are still kept in higher precision (e.g. BF16) so training remains numerically stable.
- **Quantization** — representing numbers with fewer bits (e.g. 4-bit instead of 16-bit), trading some numerical precision for a large reduction in memory footprint. This is what makes it possible to fine-tune very large models (tens of billions of parameters) on a single consumer GPU.

# PyTorch Training and Performance Concepts

## 44. Gradient Clipping

Useful for training stability, especially with recurrent models:

```python
loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
)

optimizer.step()
```

**Expected Output:** `clip_grad_norm_` returns the *original* (pre-clip) total gradient norm as a scalar tensor, e.g. `tensor(3.42)`, while modifying every parameter's `.grad` in place so their combined norm doesn't exceed `1.0`.

**Key Terms:**
- **Gradient clipping** — rescales all gradients together if their combined magnitude exceeds a threshold, preventing a single "exploding gradient" (an unusually large gradient, common in RNNs over long sequences) from taking a destructively large optimizer step.
- **`max_norm`** — the threshold on the gradient's overall L2 norm (the "length" of the gradient vector, treating all parameters as one big vector).

**Math Behind It:**
The total gradient norm is `‖g‖ = √(Σ gᵢ²)` across every parameter `gᵢ`. If `‖g‖ > max_norm`, every gradient is rescaled by `max_norm / ‖g‖`, which shrinks the gradient vector down to exactly `max_norm` in length while preserving its *direction* — so training still moves the right way, just with a bounded step size.

## 45. Gradient Accumulation

Useful when a desired batch does not fit into GPU memory:

```python
accumulation_steps = 4

optimizer.zero_grad()

for step, (x, y) in enumerate(loader):
    logits = model(x)

    loss = loss_fn(logits, y)
    loss = loss / accumulation_steps

    loss.backward()

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Effective batch size is approximately:

`physical batch size × accumulation steps`

**Key Terms:**
- **Gradient accumulation** — runs several small ("physical") batches through `backward()` *without* calling `optimizer.step()` in between, letting gradients add up (accumulate) across them, then finally takes one optimizer step using the combined gradient — simulating a much larger batch than would otherwise fit in GPU memory.
- **`loss = loss / accumulation_steps`** — since gradients from multiple `.backward()` calls simply sum, dividing the loss first ensures the *accumulated* gradient ends up equivalent to what you'd get from one big batch's *average* loss, rather than being 4× too large.

**Math Behind It:**
If physical batch size is 8 and `accumulation_steps=4`, the effective batch size the optimizer "sees" per update is `8 × 4 = 32`, even though only 8 examples' activations are ever held in GPU memory at once.

## 46. Mixed Precision

```python
with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
):
    logits = model(x)
    loss = loss_fn(logits, y)
```

Benefits include lower memory use and potentially higher GPU throughput.

BF16 is often preferred on hardware that supports it.

**Key Terms:**
- **Mixed precision** — runs most operations in a lower-precision floating point format (16-bit) instead of the default 32-bit, while automatically keeping numerically sensitive operations in 32-bit — roughly halving memory use and often speeding up matrix multiplications on modern GPUs, which have specialized hardware for 16-bit math.
- **`torch.autocast`** — a context manager that automatically picks the appropriate precision (16-bit or 32-bit) for each operation inside the block, so you don't have to manually cast every tensor.
- **FP16 vs BF16** — both are 16-bit formats, but they split their bits differently. FP16 has more precision (mantissa bits) but a smaller representable range, making it prone to numerical overflow during training. BF16 has the same exponent range as standard 32-bit floats (less precision, but much less risk of overflow), which is why it's often preferred for training when the hardware supports it.

## 47. Gradient Checkpointing

Conceptually:

```text
Normal:
forward → store many activations → backward

Checkpointing:
forward → store fewer activations
       → recompute some during backward
```

Trade-off:

`less memory ↔ more compute`

This is important for large Transformer training.

**Key Terms:**
- **Activations** — the intermediate outputs of each layer during the forward pass, which normally must be kept in memory because `backward()` needs them to compute gradients.
- **Gradient (activation) checkpointing** — instead of storing every layer's activations, it only stores a few "checkpoints" and *recomputes* the activations in between on-the-fly during the backward pass. This trades extra compute (roughly one extra forward pass) for significantly less peak memory usage, which is often the limiting factor when training very large models.

## 48. ModuleList

Use `ModuleList` when dynamically creating registered layers:

```python
self.layers = nn.ModuleList([
    TransformerBlock(...)
    for _ in range(6)
])
```

A normal Python list does not properly register contained modules.

**Key Terms:**
- **`nn.ModuleList`** — a list-like container specifically for `nn.Module` objects, which ensures PyTorch's internal machinery (e.g. `.parameters()`, `.to(device)`, `state_dict()`) correctly "sees" every module inside it.
- **Why not a plain Python `list`** — if you stored the same layers in a regular `[TransformerBlock(...) for _ in range(6)]` list, PyTorch wouldn't recognize them as sub-modules: calling `model.parameters()` would silently miss all their weights, `.to(device)` wouldn't move them, and `model.state_dict()` would be missing entries — a subtle bug that's easy to introduce.

## 49. Parameter

Explicit trainable tensor:

```python
self.weight = nn.Parameter(
    torch.randn(100, 100)
)
```

LoRA matrices are commonly represented with `nn.Parameter`.

**Key Terms:**
- **`nn.Parameter`** — a thin wrapper around a tensor that tells `nn.Module`, "this is a trainable weight — include it in `.parameters()`, move it with `.to(device)`, and save it in `state_dict()`." Layers like `nn.Linear` create their weights this way internally; you use it directly when building a custom layer with weights that don't come from an existing `nn.Module` subclass (like `lora_a`/`lora_b` in Section 36).

## 50. Register Buffer

For non-trainable state that should move with the model:

```python
self.register_buffer(
    "mask",
    torch.tril(
        torch.ones(512, 512)
    ),
)
```

Useful for causal masks and other persistent tensors.

**Key Terms:**
- **Buffer** — a tensor that's part of the model's state (moves with `.to(device)`, gets saved/loaded in `state_dict()`) but is *not* trained (it has no gradient, and the optimizer never touches it). A precomputed causal mask (Section 33) is a perfect example: it needs to live on the same device as the model, but it's a fixed constant, not a learnable weight.
- **`register_buffer(name, tensor)`** — the method that registers such a tensor, making it accessible afterward as `self.mask`, just like a regular attribute.

## 51. Detach

```python
y = x.detach()
```

Disconnects the tensor from the autograd graph.

Useful for logging, storing predictions, or intentionally stopping gradient flow.

**Key Terms:**
- **`.detach()`** — returns a new tensor that shares the same underlying data as `x`, but is disconnected from the computational graph, so no gradient will ever flow back through it. Useful when you want a tensor's *values* (e.g. for logging, plotting, or storing predictions) without accidentally keeping the entire autograd history alive in memory, or when you deliberately want to block gradients from flowing into part of a computation.

## 52. Shape Debugging

For AI interviews, shape tracking is critical.

Typical Transformer shapes:

```text
Tokens:           [B, T]
Embedding:        [B, T, D]
Q/K/V:            [B, H, T, Dh]
Attention scores: [B, H, T, T]
Context:          [B, H, T, Dh]
Merged:           [B, T, D]
```

Be able to derive these without looking them up.

**Key Terms:**
- **`B, T, D, H, Dh`** — the standard shorthand: **B**atch size, sequence length (**T**ime/Tokens), model/embedding dimension (**D**), number of attention **H**eads, and per-head dimension (**D**h `= D / H`).
- Practicing deriving this table from scratch (rather than memorizing it) is what actually helps in interviews — each row follows mechanically from the operation above it: embedding adds a `D` dimension, head-splitting divides `D` into `H × Dh` and moves `H` next to `B`, attention scores come from `Q @ Kᵀ` (so the two `T`s multiply against each other, not `Dh`), and merging reverses the head split.

# Interview Checklist

For a Senior AI Architect / GenAI Architect interview, be able to implement from memory:

1. Basic `nn.Module`
2. Training loop
3. Custom Dataset/DataLoader
4. CNN classifier
5. RNN/LSTM classifier
6. Embedding layer
7. Scaled dot-product attention
8. Multi-head attention
9. Transformer block
10. Causal attention mask
11. Small Transformer
12. LoRA linear layer
13. Freeze base model parameters
14. Count trainable parameters
15. Merge LoRA weights
16. Gradient accumulation
17. Mixed precision

Know conceptually:

* Tensor/autograd
* CNN
* RNN/LSTM
* Attention
* Transformer encoder/decoder
* Causal masking
* Positional encoding and RoPE
* AdamW
* Mixed precision
* Gradient checkpointing
* Distributed training
* LLM inference
* LoRA
* QLoRA

## What to Prioritize

For your target role, prioritize:

```text
1. Tensor + autograd
2. nn.Module + training loop
3. Embeddings
4. Attention
5. Multi-head attention
6. Transformer architecture
7. Causal masking
8. LoRA / QLoRA
9. Inference and performance
10. CNN/RNN fundamentals
```

The most important transition is:

```text
PyTorch basics
      ↓
Attention
      ↓
Transformer
      ↓
LLM
      ↓
LoRA / QLoRA
      ↓
RAG + Agents
      ↓
Production AI Architecture
```
