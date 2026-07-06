
import random
from custom_autograd.engine import Value

class Neuron:

  def __init__(self, nin):
    # initializing the weights
    self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]

    # initializing the bias
    self.bias = Value(random.uniform(-1, 1))


  def __call__(self, x):
    # multiplying every input by it's corresponding weight
    bi = self.bias
    for wi, xi in zip(self.weights, x):
      bi = bi + wi * xi
    return bi.tanh()


  def parameters(self):
    # returning model parameters
    return self.weights + [self.bias]


class Layer:

  def __init__(self, nin, nout):
    # Create a list of 'nout' distinct neurons
    self.neurons = [Neuron(nin) for _ in range(nout)]


  def __call__(self, x):
    # Passing input list x to every single neuron and making a list of their outputs
    outputs = [neuron(x) for neuron in self.neurons]

    # single neuron - return salar value object, multi neuron - return output list
    return outputs


  def parameters(self):
    # store weights and bais for each neuron
    params = []
    for neuron in self.neurons:
      params.extend(neuron.parameters())
    return params


class MLP:

  def __init__(self, nin, nouts):
    # 1. Combine input dimensions with output dimensions
    sizes = [nin] + nouts

    # 2. Build the layers sequentially by mapping adjacent sizes
    self.layers = []
    for i in range(len(nouts)):
      self.layers.append(Layer(sizes[i], sizes[i+1]))


  def __call__(self, x):
    # Feeds the input vector sequentially through every single layer
    for layer in self.layers:
      x = layer(x)
    return x[0]


  def parameters(self):
    params = []
    for layer in self.layers:
      params.extend(layer.parameters())
    return params
