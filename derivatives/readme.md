# 📉 Accumulation Principles & Base Computational Graph Mechanics

**Source Material Reference:** `01_chain_rule_addition.jpeg`  
**Target Engine Implementations:** `custom_autograd/engine.py`

This document details the first-principles calculus foundations for the core graph-walking engine sketched out during the structural design phase. These equations map how the dynamic operational tape compiles dependencies, resolves variable caching, and updates scalar nodes backward through complex topological pathways.

---

## 1. The Global Tracking Node Topology

To execute algorithmic backpropagation without deep-learning frameworks, individual mathematical operations are treated as discrete coordinate nodes within a dynamically compiled **Directed Acyclic Graph (DAG)**.

Every computational operation creates an explicit child-to-parent reference pair. When evaluating an upstream expression, the engine preserves the incoming mathematical variables within an internal structural collection (`_children`) and links them to the generative operational identifier (`_op`).

---

## 2. Dynamic Backpropagation via the Multivariate Chain Rule

The foundational objective of the autograd engine is compiling the total partial derivative of an arbitrary root optimization scalar (Global Loss, $L$) with respect to every single underlying weight, bias, and input coordinate.

### Mathematical Derivation
Let an active operational variable $\text{self}$ feed into an immediate downstream variable $\text{out}$ during the forward execution pass. By the single-variable Chain Rule definition, the partial derivative updates as:

$$\frac{\partial L}{\partial \text{self}} = \frac{\partial L}{\partial \text{out}} \times \frac{\partial \text{out}}{\partial \text{self}}$$

However, in multi-layer topologies, an independent variable can branch out and contribute to multiple parallel downstream execution expressions ($y_1, y_2, \dots, y_i$). To prevent gradient leakage and account for all concurrent pathways, the engine implements the full **Multivariate Chain Rule**:

$$\frac{\partial L}{\partial \text{self}} = \sum_{i} \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial \text{self}}$$

### The Accumulation Constraint (`+=`)
This multivariate property mathematically mandates the use of accumulation assignments (`+=`) rather than standard static assignments (`=`) inside the inner `_backward` execution routines:

$$g_{\text{self}} \leftarrow g_{\text{self}} + g_{\text{out}} \cdot \delta_{\text{local}}$$

Where $g$ represents the compiled node gradient (`.grad`) and $\delta_{\text{local}}$ represents the isolated local derivative.

```python
# Core Engine Vector Integration (`engine.py`):
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0          # Initialized to zero accumulation
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

# 🔬 Analytical Atomic Overloads & Activation Parameter Derivatives

**Source Material Reference:** `02_tanh_power_derivatives.jpeg`  
**Target Engine Implementations:** `custom_autograd/engine.py`, `custom_autograd/nn.py`

This document details the first-principles calculus derivations for non-linear activations and non-linear scalar overloads sketched out during the core design phase. These equations map out the exact local gradient multipliers required to backpropagate scalar evaluation chains through deep architectures without numerical collapse.

---

## 1. The Hyperbolic Tangent ($\tanh$) Non-Linearity

The Multi-Layer Perceptron uses the hyperbolic tangent function to scale hidden neuron activations into a bounded symmetric range $[-1.0, 1.0]$. 

### Mathematical Derivation
Let the forward activation node be defined as $out = \tanh(\text{self})$. Using standard differential quotients:

$$\frac{d}{dx}\tanh(x) = \text{sech}^2(x) = 1 - \tanh^2(x)$$

Applying the Multivariate Chain Rule to propagate the global loss scalar ($L$) upstream yields:

$$\frac{\partial L}{\partial \text{self}} = \frac{\partial L}{\partial \text{out}} \times \frac{\partial \text{out}}{\partial \text{self}}$$

$$\frac{\partial L}{\partial \text{self}} = \frac{\partial L}{\partial \text{out}} \times \left(1 - \tanh^2(\text{self})\right)$$

Because $out = \tanh(\text{self})$ is already computed during the forward pass and stored in heap memory, we bypass costly re-evaluation of the transcendental function during backpropagation by substituting the cached `out.data` property:

$$\frac{\partial L}{\partial \text{self}} = \frac{\partial L}{\partial \text{out}} \times \left(1 - out^2\right)$$

```python
# System Engine Implementation (`engine.py`):
def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, _children=(self,), _op='tanh')
    
    def _backward():
        # Gradient computation utilizing cached forward data
        self.grad += out.grad * (1.0 - t**2)
    out._backward = _backward
    
    return out
```
