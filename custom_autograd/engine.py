import math

_TRACK_GRADIENTS = True

class no_grad:
  def __enter__(self):
    global _TRACK_GRADIENTS
    self.prev = _TRACK_GRADIENTS
    _TRACK_GRADIENTS = False

  def __exit__(self, *args):
    global _TRACK_GRADIENTS
    _TRACK_GRADIENTS = self.prev

class Value:
  def __init__(self, data, _children=(), _op=""):
    self.data = float(data) # stores data

    self.grad = 0.0 # stores gradient

    self._prev = set(_children) # set of parent nodes that created current node

    self._op = _op if _TRACK_GRADIENTS else ""# operation used to create current node

    self._backward = lambda: None # buffer backward function that will be replaced


  @staticmethod
  def _autowrap(other):
    return other if isinstance(other, Value) else Value(other)


  def __add__(self, other):
    # 1. Constanct fallback(Autowrapping)
    # check if other is a Value object
    other = self._autowrap(other)

    # 2. Forward Pass:
    out = Value(self.data+other.data, _children=(self, other), _op="+")

    # 3. Local backward function
    def _backward():
      self.grad += out.grad * 1.0
      other.grad += out.grad * 1.0

    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out


  def __radd__(self, other):
    return self + other


  def __mul__(self, other):
    # 1. Autowrapping
    # check if other is instance of Value class
    other = self._autowrap(other)

    # 2. Forward pass
    out = Value(self.data*other.data, _children=(self, other), _op="*")

    # 3. Backward pass
    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __rmul__(self, other):
    return self * other


  def __pow__(self, other):
    assert isinstance(other, (int, float))

    x = self.data
    y = other
    out = Value(x**y, _children=(self,), _op="**")

    def _backward():
      self.grad += out.grad * (y*x**(y-1))

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __rpow__(self, other):
    return self._autowrap(other)**self


  def __sub__(self, other):
    other = self._autowrap(other)

    out = Value(self.data-other.data, _children=(self, other), _op="-")

    def _backward():
      self.grad += out.grad * 1.0
      other.grad += out.grad * (-1.0)

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __rsub__(self, other):
    return self._autowrap(other) - self


  def __truediv__(self, other):
    other = self._autowrap(other)

    x, y = self.data, other.data

    out = Value(x/y, _children=(self, other), _op="/")

    def _backward():
      self.grad += out.grad * 1/y
      other.grad += out.grad * (x*-1.0*(y**-2))

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __rtruediv__(self, other):
    return self._autowrap(other) / self


  def __neg__(self):
    out = Value(-self.data, _children=(self,), _op="neg")

    def _backward():
      self.grad += out.grad * -1.0

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __rneg__(self):
    return -self


  def backward(self):
    # 1. Custom DFS Graph sort
    topo = []
    visited = set()

    def dfs(node):
      if node not in visited:
        visited.add(node)

        for parent in node._prev:
          dfs(parent)
        topo.append(node)
    dfs(self)

    # 2. Set the initial output gradient to 1.0
    self.grad = 1.0

    # 3. Walk the sorted graph in reverse
    for node in reversed(topo):
      node._backward()


  def tanh(self):
    next_h = math.tanh(self.data)
    out = Value(next_h, _children=(self,), _op="tanh")

    def _backward():
      self.grad += out.grad * (1-next_h*next_h)

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def relu(self):
    x = max(0, self.data)
    out = Value(x, _children=(self,), _op="relu")

    def _backward():
      self.grad += out.grad * (1.0 if x>0 else 0.0)

    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __repr__(self):
    # function for custom representation used for debugging.
    return f"Value(data={self.data}, grad={self.grad})"
