# Tiny Autograd Engine and MLP Implementation

This project implements a simple autograd engine and a Multilayer Perceptron (MLP) from scratch, using Python. The code is educational and showcases the fundamentals of automatic differentiation, neural networks, and backpropagation.

---

## Features

1. *Autograd Engine*:
   - Implements a custom Value class to store data, gradients, and compute operations.
   - Supports key operations (+, *, /, -, exp, pow, tanh, ReLU).
   - Includes a backward pass for automatic differentiation.

2. *Neural Network Components*:
   - Neuron: Represents a single neuron with weights, bias, and activation function (ReLU/linear).
   - Layer: Represents a layer of neurons.
   - MLP: Constructs a Multilayer Perceptron (MLP) with configurable architecture.

3. *Visualization*:
   - Generates a computational graph using graphviz.

4. *Example Usage*:
   - Define and train an MLP on a toy dataset using custom forward and backward passes.

---

## How It Works

### Autograd Engine

The Value class serves as the core data structure, enabling:
- Storage of data and gradients.
- Construction of computation graphs.
- Backpropagation for gradient computation via the backward method.

### MLP Architecture

The MLP class uses:
- *Layers* composed of *Neurons*.
- Configurable architecture specified as a list of layer sizes.

---

## Code Example

```python
# Define inputs and targets
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Initialize MLP with input size 3 and hidden layers [4, 4, 1]
n = MLP(3, [4, 4, 1])

# Forward pass to compute predictions
ypred = [n(x) for x in xs]

# Compute loss (Mean Squared Error)
loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])

# Backpropagation
loss.backward()

# Visualize the computation graph
from graphviz import Digraph
draw_dot(loss).render("computation_graph", format="svg")
```
