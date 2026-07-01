# 🧠 Core Mathematical Foundations & Analytical Derivations

This directory preserves the raw, first-principles mathematical derivations executed on paper during the architectural planning of this object-oriented automatic differentiation engine. 

Before translating graph-walking algorithms into Python objects, individual partial differential equations were explicitly mapped out to verify how local gradients propagate through atomic operational nodes using reverse-mode automatic differentiation.

---

## 1. Foundational Architecture & The Multivariate Chain Rule
**Source Material:** `01_chain_rule_addition.jpeg`

The framework evaluates the global optimization objective gradient by mapping the **Multivariate Chain Rule** onto the dynamically constructed Directed Acyclic Graph (DAG). 

For a scalar root optimization objective $L$ and an arbitrary scalar node $\text{self}$ contributing to an execution path leading to a downstream dependent node $\text{out}$, the fundamental backward propagation equation resolves as:

$$\frac{\partial L}{\partial \text{self}} = \frac{\partial L}{\partial \text{out}} \times \frac{\partial \text{out}}{\partial \text{self}}$$

In the custom codebase, this rule is translated directly into cumulative gradient updates within each node's internal `_backward` execution tape:

$$\text{self.grad} \mathrel{+}= \text{out.grad} \times \text{local\_gradient}$$

### Case Study: Linear Accumulation via Addition Overloads
When an addition node is compiled ($out = \text{self} + \text{other}$), the localized partial derivative with respect to each independent variable isolates perfectly to a constant unit value:

$$\frac{\partial \text{out}}{\partial \text{self}} = \frac{\partial}{\partial \text{self}}(\text{self} + \text{other}) = 1.0$$

$$\frac{\partial \text{out}}{\partial \text{other}} = \frac{\partial}{\partial \text{other}}(\text{self} + \text{other}) = 1.0$$

Thus, the addition layer behaves as a pure **gradient pass-through filter**, distributing the incoming downstream loss signal evenly to ancestral parent variables without scaling alterations:

```python
# Code Translation inside engine.py:
def _backward():
    self.grad  += out.grad * 1.0
    other.grad += out.grad * 1.0
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
