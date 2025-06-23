# Autograd

A minimal, scalar-valued automatic differentiation engine written in Python.  
This project implements a custom `Node` class to build and backpropagate through mathematical expressions, much like PyTorch's autograd — but from scratch and educationally.

---

## Overview

`Autograd` allows you to:

- Create scalar mathematical expressions using overloaded operators (`+`, `*`, `**`, etc.)
- Track how values were computed using a dynamic computation graph.
- Perform automatic differentiation via a `.backward()` call.
- Visualize how gradients flow using the chain rule.

---

## Features

- Operator overloading for expressions: `+`, `-`, `*`, `/`, `**`
- Activation functions: `ReLU`, `tanh`
- Gradient computation using reverse-mode automatic differentiation
- Dependency tracking through computation graph

---

## How It Works

### `Node` Class

Each `Node` represents a value in your computation graph and stores:

- `data`: the actual scalar value
- `grad`: the computed gradient after `.backward()`
- `prev`: a set of input nodes (children) used to create this node
- `op`: a string representing the operation used (e.g. `+`, `*`, `relu`)
- `_backward`: a function that applies the chain rule to this node
- `label` *(optional)*: for naming nodes in debugging or visualization

### `children` and `prev`

The `children` are passed in as a tuple for convenience, but stored as a set in `.prev`. This lets us keep track of which previous nodes led to a specific output, allowing proper backpropagation without duplicates.

### `operation`

This string is purely descriptive — it stores **what mathematical operation** was used to generate this node’s value. This is useful for graph visualization or debugging, since the value alone doesn’t tell us how it was derived.

### `_backward()` Function

Each node stores a `_backward()` function — its own piece of the **chain rule**. When `.backward()` is called on the final node, it:

1. Performs a topological sort of the graph.
2. Starts from the final output node (whose gradient is set to 1).
3. Recursively calls `_backward()` on each node to accumulate gradients flowing backward.

---

## Example

```python
from src.autograd import Node

x = Node(2.0, label='x')
y = Node(3.0, label='y')
z = x * y + x**2 + y.relu()
z.backward()

print(f"z: {z.data}")
print(f"dz/dx: {x.grad}")
print(f"dz/dy: {y.grad}")
