
import math

_TRACK_GRADIENTS = True # flag for gradient tracking

# Class no_grad to switch off gradient tracking during inference
class no_grad:

  def __enter__(self):
    global _TRACK_GRADIENTS
    self.prev = _TRACK_GRADIENTS
    _TRACK_GRADIENTS = False


  def __exit__(self, *args):
    global _TRACK_GRADIENTS
    _TRACK_GRADIENTS = self.prev


# Class Value to handle main engine calculus and backpropagation
class Value:

  def __init__(self, data, _children=(), _op=""):
    self.data = float(data) # stores data

    self.grad = 0.0 # stores gradient

    self._prev = set(_children) # set of parent nodes that created current node

    self._op = _op if _TRACK_GRADIENTS else ""# operation used to create current node

    self._backward = lambda: None # buffer backward function that will be replaced

    # static replay pointer cache
    self._compiled_topology = None
    self._forward_replay = lambda: None


  @staticmethod
  def _autowrap(other):
    # autowraps scalar value to Value class object
    return other if isinstance(other, Value) else Value(other)


  def _build_topo(self):
    # builds custom graph topology for backpropagation
    topo = []
    visited = set()

    def dfs(node):
      if node not in visited:
        visited.add(node)
        for parent in node._prev:
          dfs(parent)
        topo.append(node)
    dfs(self)
    return topo


  def compile_topology(self):
    # using _build_topo function
    return self._build_topo()


  def __add__(self, other):
    # 1. constanct fallback(Autowrapping)
    # check if other is a Value object
    other = self._autowrap(other)

    # 2. forward Pass:
    out = Value(self.data+other.data, _children=(self, other), _op="+")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * 1.0
      other.grad += out.grad * 1.0

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = self.data + other.data
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def __radd__(self, other):
    # reverse addition
    return self + other


  def __mul__(self, other):
    # 1. autowrapping
    other = self._autowrap(other)

    # 2. forward pass
    out = Value(self.data*other.data, _children=(self, other), _op="*")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = self.data * other.data
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def __rmul__(self, other):
    # reverse multiplication
    return self * other


  def __pow__(self, other):
    # checking if other is an integer or float value
    assert isinstance(other, (int, float))

    # 2. forward pass
    out = Value(self.data**other, _children=(self,), _op="**")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * (other*self.data**(other-1))

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = self.data**other
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def __rpow__(self, other):
    # reverse power function
    return self._autowrap(other)**self


  def __sub__(self, other):
    # 1. autowrapping
    other = self._autowrap(other)

    # 2. forward pass
    out = Value(self.data-other.data, _children=(self, other), _op="-")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * 1.0
      other.grad += out.grad * (-1.0)

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = self.data - other.data
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def __rsub__(self, other):
    # reverse subtraction
    return self._autowrap(other) - self


  def __truediv__(self, other):
    # 1. autowrapping
    other = self._autowrap(other)

    # 2. forward pass
    out = Value(self.data/other.data, _children=(self, other), _op="/")

    # 3. local backward pass
    def _backward():
      self.grad += out.grad * 1/other.data
      other.grad += out.grad * (self.data*-1.0*(other.data**-2))

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = self.data / other.data
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def __rtruediv__(self, other):
    # reverse division
    return self._autowrap(other) / self


  def __neg__(self):
    # 2. forward pass
    out = Value(-self.data, _children=(self,), _op="neg")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * -1.0

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = -self.data
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def __rneg__(self):
    # reverse negation
    return -self


  def backward(self, compiled_path=None):
    # using compiled path for backpropagation
    if compiled_path is not None:
      self.grad = 1.0
      for node in reversed(compiled_path):
        node._backward()
      return

    # fallback to full dynamic sorting if not pre compiled
    # 1. custom DFS Graph sort
    topo = self._build_topo()

    # 2. set the initial output gradient to 1.0
    self.grad = 1.0

    # 3. walk the sorted graph in reverse
    for node in reversed(topo):
      node._backward()


  def tanh(self):
    # 2. forward pass
    next_h = math.tanh(self.data)
    out = Value(next_h, _children=(self,), _op="tanh")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * (1-out.data*out.data)

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = math.tanh(self.data)
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward

    return out # returning the resulting Value object


  def relu(self):
    # 2. forward pass
    x = max(0, self.data)
    out = Value(x, _children=(self,), _op="relu")

    # 3. local backward function
    def _backward():
      self.grad += out.grad * (1.0 if out.data>0.0 else 0.0)

    # 4. local forward replay function for backpropagation using static cache
    def _forward_replay():
      out.data = max(0.0, self.data)
    out._forward_replay = _forward_replay

    # 5. toogling gradient tracking for backpropagation
    if _TRACK_GRADIENTS:
      out._backward = _backward
    return out


  def __repr__(self):
    # function for custom representation used for debugging.
    return f"Value(data={self.data}, grad={self.grad})"


# Class FastCompiledTopology for static sped up engine nodes topology
class FastCompiledTopology:

  def __init__(self, loss_node, input_placeholders, target_placeholders):
    self.loss_node = loss_node # loss node

    self.input_placeholders = input_placeholders # input placeholders or data values
    self.target_placeholders = target_placeholders # target placeholders or target values

    self.topo_order = loss_node._build_topo() # stores the topology order using _build_topo to build static topology

    self.backward_order = [node for node in self.topo_order if len(node._prev)>0] # stores only nodes that has parents, implements dead end pruning


  def update_batch(self, batch_x, batch_y):
    # in place updates for training and targets
    for placeholder_row, sample_x in zip(self.input_placeholders, batch_x):
      for node, val_x in zip(placeholder_row, sample_x):
        node.data = float(val_x)

    for placeholder_node, val_y in zip(self.target_placeholders, batch_y):
      placeholder_node.data = float(val_y)


  def forward(self):
    # replaying the forward pass
    for node in self.topo_order:
      node._forward_replay()
    return self.loss_node.data


  def backward(self, parameters):
    # performing/replaying backward pass
    for node in self.topo_order:
      node.grad = 0.0
    for p in parameters:
      p.grad = 0.0

    self.loss_node.grad = 1.0
    for node in reversed(self.backward_order):
      node._backward()


  def predict(self, val_x):
    # only for inference purposes
    for placeholders_row, sample_x in zip(self.input_placeholders, val_x):
      for node, val in zip(placeholders_row, sample_x):
        node.data = float(val)

    # forward pass
    for node in self.topo_order:
      node._forward_replay()

    return self.loss_node.data # returns loss
