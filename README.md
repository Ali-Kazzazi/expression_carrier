# EXIER(EXPRESSION-CARRIER) — Scalar Autograd Engine & Neural Network Library in C++

A from-scratch implementation of a scalar-valued automatic differentiation engine and multilayer perceptron (MLP) framework in modern C++17, inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy. Trains on MNIST with cross-entropy loss and the Adam optimizer — no external ML libraries required.

---

## What is this?

EXIER(EXPRESSION-CARRIER) is a lightweight, educational deep learning engine built entirely from first principles. Every forward pass builds a dynamic computation graph; every backward pass traverses it to accumulate exact gradients via reverse-mode autodiff (backpropagation). The goal is to make the internals of neural network training fully transparent and readable.

---

## Project Structure

```
.
├── include/
│   ├── exier.hpp        # Core autograd engine (EXIER node, operators, activations)
│   ├── mlp.hpp          # Neuron, Layer, MLP — neural network building blocks
│   ├── loss.hpp         # MSE, MAE, Cross-Entropy, Binary Cross-Entropy losses
│   ├── optimizer.hpp    # Gradient Descent and Adam optimizers
│   └── mnist.hpp        # CSV-based MNIST data loader
├── main.cpp             # Training loop — 3-layer MLP on MNIST
├── exier_graph_visualizer.html  # Browser-based computation graph visualizer
└── CMakeLists.txt
```

---

## Core Components

### `exier.hpp` — The Autograd Engine

The `EXIER` class is the fundamental unit of computation. Each node stores:

- `data` — the scalar forward value
- `grad` — the accumulated gradient from backprop
- `prev` — parent nodes in the computation graph
- `op` — the operation that produced this node (for debugging/visualization)
- `_backward` — a closure that propagates gradients to parent nodes

**Supported operations (all differentiable):**

| Operation          | Notes                                           |
| ------------------ | ----------------------------------------------- |
| `+`, `-`, `*`, `/` | Arithmetic with EXIER or `float` on either side |
| `pow(x, n)`        | Scalar and EXIER exponents                      |
| `tanh(x)`          | Hyperbolic tangent                              |
| `exp(x)`           | Natural exponential                             |
| `log(x)`           | Natural logarithm                               |
| `softmax(vec)`     | Numerically stable (subtracts max before exp)   |

`backward()` performs a topological sort of the graph and calls each node's `_backward` closure in reverse order — implementing full reverse-mode autodiff.

---

### `mlp.hpp` — Neural Network Layers

Built on top of `EXIER`, three composable classes:

- **`Neuron`** — a single neuron with weights, bias, and a configurable activation function
- **`Layer`** — a collection of neurons forming one fully-connected layer
- **`MLP`** — stacks layers into a complete multilayer perceptron

**Available activations:**

```cpp
act_tanh()      // Hyperbolic tangent
act_relu()      // Rectified Linear Unit
act_sigmoid()   // Sigmoid / logistic
act_linear()    // Identity (no activation)
```

Example — defining a 3-layer network:

```cpp
auto model = MLP(784, {128, 64, 10}, {act_relu(), act_relu(), act_linear()});
```

---

### `loss.hpp` — Loss Functions

| Function                            | Use case                                                |
| ----------------------------------- | ------------------------------------------------------- |
| `mse_loss(pred, target)`            | Regression, one-hot MSE                                 |
| `mae_loss(pred, target)`            | Mean absolute error                                     |
| `cross_entropy_loss(logits, class)` | Multi-class classification (applies softmax internally) |
| `bce_loss(pred, target)`            | Binary classification                                   |

All losses return an `EXIER` node — fully differentiable, ready for `.backward()`.

---

### `optimizer.hpp` — Optimizers

**Gradient Descent (`GD`):**
```cpp
GD optimizer(params.size(), /*lr=*/0.01f);
optimizer.step(params);
```

**Adam (`Adam`):**
```cpp
Adam optimizer(params.size(), /*lr=*/0.001f);
optimizer.step(params);
```

Adam implements first and second moment estimates with bias correction — numerically stable and well-suited for sparse gradients and noisy loss landscapes.

---

### `mnist.hpp` — Data Loader

Reads MNIST data from CSV format (compatible with the standard Kaggle MNIST CSVs). Provides two loading modes:

- `next(exier_vector&)` — loads pixels as `shared_ptr<EXIER>` nodes, ready for direct model input
- `next(sample_vector&)` — loads pixels as raw `float` vectors for preprocessing

Pixels are automatically normalized to `[0, 1]` by dividing by 255.

---

### `exier_graph_visualizer.html` — Graph Visualizer

A standalone browser tool to inspect computation graphs. Your C++ program can output the graph as JSON via `node->toJSON()`, which you then paste or drag-drop into the visualizer to render an interactive node graph — useful for debugging forward/backward passes on small expressions.

---

## Training Example (MNIST)

`main.cpp` trains a `784 → 128 → 64 → 10` MLP on the first 200 MNIST samples using mini-batch cross-entropy loss and Adam:

```cpp
auto model = MLP(784, {128, 64, 10}, {act_relu(), act_relu(), act_linear()});
auto params = model.parameters();
Adam optimizer(params.size(), 0.001f);

// Mini-batch training loop
for (int epoch = 1; epoch <= 200; epoch++) {
    // ... forward, loss, backward, step
}
```

Output per epoch:
```
epoch 1 | loss = 2.3041 | acc = 18/200
epoch 2 | loss = 2.1872 | acc = 34/200
...
```

---

## Build

```bash
cmake --build build --config Release

.\bin\Release\exier.exe or ./bin/exier

```

Requires: C++17, CMake ≥ 3.20. No third-party dependencies.

Place `mnist_train.csv` in the project root before running. The standard Kaggle MNIST CSV format is expected (first column = label, remaining 784 columns = pixel values).

---

## Design Advantages

- **Zero dependencies** — pure C++17 standard library, no Eigen, no PyTorch, no BLAS
- **Transparent gradients** — every gradient computation is a readable lambda closure, not a black box
- **Dynamic computation graph** — the graph is rebuilt each forward pass; supports arbitrary control flow
- **Memory-safe** — uses `shared_ptr` throughout; weak pointers in backward closures prevent retain cycles
- **Composable** — loss functions, optimizers, and layers are all independent and interchangeable
- **Debuggable** — `repr()` prints the graph as an ASCII tree; `toJSON()` exports it for visual inspection
- **Numerically stable softmax** — subtracts max before exponentiation to avoid overflow

---

## Limitations

This is a scalar engine — each number is its own graph node. This makes it educational and exact, but slow for large networks compared to tensor-based frameworks that operate on batched matrix operations. It is best suited for learning, experimentation, and small-scale problems.