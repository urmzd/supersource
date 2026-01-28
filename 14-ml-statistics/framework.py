# Extracted from framework.ipynb
# NOTE: Code cells preserved; markdown condensed into section headers/comments

# ==============================================================================
# Section 1: Notebook
# ==============================================================================

# <a href="https://colab.research.google.com/github/urmzd/school/blob/main/year-four/CSCI4155/assignments/a1/Assignment1-Students.ipynb" target="parent"<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/</a

# ==============================================================================
# Section 2: Assignment 1
# ==============================================================================

# %% Cell 0
from google.colab import drive
drive.mount('/content/drive')

# In this assignment you will be implementing some components of neural networks in numpy from scratch (all your code should be vectorized; and you shouldn’t use any library besides numpy and matplotlib). Base meta classes for the various activations, layers and optimizers are provided; all of your implementations must be classes that inherit from the appropriate base meta class. When implementing backwards passes, please include in Markdown an analytical derivation of the backwards pass.

# ==============================================================================
# Section 3: Step 1: Fully-connected neural networks
# ==============================================================================

# ==============================================================================
# Section 4: General comments
# ==============================================================================

# %% Cell 1
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# Section 5: 1.
# ==============================================================================

# ==============================================================================
# Section 6: ReLu
# ==============================================================================

# %% Cell 2
import abc

# Abstract base class for all activation classes
class Activations(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def backward(self, grad, original_input):
        return


class ReLU(Activations):
    def forward(self, x):
        # return result
        return np.maximum(0, x)

    def backward(self, grad, original_input):
        # this assumes that the original input to this layer has been saved somewhere else
        x = original_input
        return grad * (x > 0)

    def __call__(self, x, mode=None):
        return self.forward(x)


class Sigmoid(Activations):
    def _activate(self, x: np.ndarray):
        divisor = 1 + np.power(np.e, -x)
        return np.reciprocal(divisor)

    def forward(self, x):
        return self._activate(x)

    def backward(self, grad, original_input):
        sigma = self._activate(original_input)
        activation = sigma * (1 - sigma)
        cost_wrt_activation = activation * grad

        return cost_wrt_activation

    def __call__(self, x, mode=None):
        return self.forward(x)


class Tanh(Activations):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, grad, original_input):
        activation = 1 - np.square(np.tanh(original_input))
        cost_wrt_activation = activation * grad

        return cost_wrt_activation

    def __call__(self, x, mode=None):
        return self.forward(x)

# ==============================================================================
# Section 7: 2.
# ==============================================================================

# ==============================================================================
# Section 8: Softmax
# ==============================================================================

# %% Cell 3
class SoftMaxCrossLoss(Activations):
    def __init__(self):
        self.probs = None
        self.y = None

    def _stable_soft_max(self, x, axis=1):
        shifted_x = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted_x)
        z = exp / np.sum(exp, axis=axis, keepdims=True)

        return z

    def forward(self, x, y=None):
        z = self._stable_soft_max(x)

        if y is None:
            return z

        loss = -np.log(z, where=(z != 0)) * y

        self.y = y
        self.probs = z

        return np.average(loss)

    def backward(self, grad, original_input):
        return self.probs - self.y

    def __call__(self, x, y=None, mode=None):
        return self.forward(x, y)

# ==============================================================================
# Section 9: 3.
# ==============================================================================

# ==============================================================================
# Section 10: Affine Layer
# ==============================================================================

# %% Cell 4
class Layers(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def backward(self, grad, original_input):
        return


class AffineLayer(Layers):
    def __init__(self, input_dim, hidden_units):
        ran = np.sqrt(1 / input_dim)

        # Initializing the weights and bias, in that order
        self.parameters = [
            np.random.uniform(-ran, ran, (hidden_units, input_dim)),
            np.random.uniform(-ran, ran, hidden_units),
        ]

        self.grads = [
            np.zeros_like(self.parameters[0]),
            np.zeros_like(self.parameters[1]),
        ]

    def forward(self, x):
        w, b = self.parameters
        return x @ w.T + b

    def backward(self, grad, original_input):
        w, _ = self.parameters

        dx = grad @ w
        dw = grad.T @ original_input
        db = np.sum(grad, axis=0)

        self.grads[0] = dw
        self.grads[1] = db

        return dx

    def __call__(self, x, mode=None):
        return self.forward(x)


class Dropout(Layers):
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x, mode):
        if mode == "train":
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p
            return self.mask * x
        else:
            return self.mask * x

    def backward(self, grad, original_input):
        return grad * self.mask

    def __call__(self, x, mode="test"):
        return self.forward(x, mode)

# ==============================================================================
# Section 11: 4.
# ==============================================================================

# %% Cell 5
class FCNN:
    def __init__(self, layers):
        self.layers = layers
        self.original_inputs = []

    def forward(self, x, y=None, mode="test"):
        self.original_inputs = [x.copy()]

        for layer in self.layers:
            prev_output = self.original_inputs[-1]
            new_output = (
                layer(x=prev_output, mode=mode, y=y)
                if isinstance(layer, SoftMaxCrossLoss)
                else layer(x=prev_output, mode=mode)
            )

            self.original_inputs.append(new_output)

        return self.original_inputs[-1]

    def backward(self):
        # Pop loss value from the cache
        loss = self.original_inputs.pop()
        grad = None

        L = len(self.layers)

        for l in range(L - 1, 0, -1):
            layer = self.layers[l]
            inp = self.original_inputs.pop()
            grad = layer.backward(grad=grad, original_input=inp)

    def __call__(self, x, y=None, mode="test"):
        return self.forward(x, y, mode)

# ==============================================================================
# Section 12: 5.
# ==============================================================================

# %% Cell 6
class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, layer):
        return


class SGD(Optimizer):
    def __init__(self, momentum, alpha=0.001):
        self.momentum = momentum
        self.prev_m_b = None
        self.prev_m_w = None
        self.alpha = alpha

    def step(self, layer):
        _, b = layer.parameters
        w_grad, b_grad = layer.grads

        batch_no = np.random.randint(b.shape[0])
        
        w_grad_random_index = w_grad[batch_no]
        b_grad_random_index = b_grad[batch_no]

        if self.prev_m_w is None and self.prev_m_b is None:
          self.prev_m_w = self.alpha * w_grad_random_index
          self.prev_m_b = self.alpha * b_grad_random_index
        else:
            self.prev_m_w = self.momentum * self.prev_m_w + self.alpha * w_grad_random_index
            self.prev_m_b = self.momentum * self.prev_m_b + self.alpha * b_grad_random_index

        layer.parameters[0][batch_no] -= self.prev_m_w
        layer.parameters[1][batch_no] -= self.prev_m_b

    def __call__(self, layer):
        self.step(layer)

# ==============================================================================
# Section 13: 6.
# ==============================================================================

# %% Cell 7
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def load_data():
    # Update prefix to be nothing on local machine.
    prefix="/content/drive/MyDrive/"
    train_data = np.load(prefix + "train.npz")["arr_0"]
    train_targets = np.load(prefix +"train.npz")["arr_1"]

    test_data = np.load(prefix + "test.npz")["arr_0"]
    test_targets = np.load(prefix + "test.npz")["arr_1"]

    y_encoder = None

    def and_reshape(train_mode=True):
        x, y = (train_data, train_targets) if train_mode else (test_data, test_targets)

        x_reshaped = x.reshape(-1, x.shape[1] ** 2)
        y_reshaped = y.reshape(-1, 1)

        nonlocal y_encoder
        if not y_encoder:
            y_encoder = OneHotEncoder()
            y_encoder.fit(y_reshaped)
            y_reshaped = y_encoder.transform(y_reshaped).toarray()

        return x_reshaped, y_reshaped, y_encoder

    return and_reshape

# %% Cell 8
import time

np.seterr("ignore")


def get_training_and_testing_accuracies_and_plot(layers, batch_size=32, n_epochs=4, mu=0.99):

    layers.insert(0, AffineLayer(784, batch_size))
    layers.append(SoftMaxCrossLoss())

    print(list(map(lambda layer: type(layer), layers)))

    optimizers = []
    for layer in layers:
      if isinstance(layer, AffineLayer):
        optimizers.append((SGD(mu), layer))

    print("-----------------------------------------------------")
    print()
    print()

    data_loader = load_data()
    x_train, y_train, encoder = data_loader(train_mode=True)

    neural_net = FCNN(layers)

    train_start = 0
    train_end = train_start + batch_size

    
    start_time = time.time()

    total_error = []
    for epoch in range(n_epochs):

      losses = []
      while train_start < y_train.shape[0]:
          # Forward Prop
          x_batch = x_train[train_start:train_end]
          y_batch = y_train[train_start:train_end]
          loss = neural_net.forward(x_batch, y_batch, mode="train")
          losses.append(loss)

          # Back Prop
          neural_net.backward()

          # Parameter Update
          for optimizer in optimizers:
              optimizer[0](optimizer[1])

          # Next Batch
          train_start += batch_size
          train_end += batch_size


      # RESTART
      train_start = 0
      train_end = batch_size
      total_error.append(np.average(np.array(losses)))

    end_time = time.time()
    print(f"TRAINING TIME FOR {n_epochs} EPOCHS: {end_time - start_time}")
    print(f"AVERAGE TRAINING LOSS: {np.average(np.array(total_error))}")

    _, ax = plt.subplots()

    ax.set_title(
        f"Average Error Per Epoch, Batch Size = {batch_size}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Error")
    ax.plot(total_error)

    x_test, y_test, _ = data_loader(train_mode=False)
    testing_accuracies = []

    test_start = 0
    test_end = batch_size

    start_time = time.time()

    while test_start < y_test.shape[0]:
        x_batch = x_test[test_start:test_end]
        y_batch = y_test[test_start:test_end]
        probs = neural_net.forward(x_batch)

        y_pred = np.argmax(probs, axis=1).reshape(-1, 1)
        accuracy = accuracy_score(y_batch, y_pred)
        testing_accuracies.append(accuracy)

        test_start += batch_size
        test_end += batch_size

    end_time = time.time()

    print(f"TESTING TIME: {end_time - start_time}")
    print(
        f"AVERAGE TESTING ACCURACY: {np.array(testing_accuracies).mean()}"
    )

# %% Cell 9
batch_sizes = [8 * i for i in range(1, 4)]

for batch_size in batch_sizes:
  layers=[
        AffineLayer(batch_size, 128), 
        ReLU(), 
        AffineLayer(128, 10), 
        ]
  get_training_and_testing_accuracies_and_plot(batch_size=batch_size, layers=layers)

# ==============================================================================
# Section 14: 6.a) Effect of Varying Batch Sizes
# ==============================================================================

# %% Cell 10
hidden_units_2_layer = [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)]
hidden_units_3_layer = [(32, 64, 64), (64, 128, 128), (128, 256, 256), (256, 512, 256), (512, 1024, 512)]

print("----------------------------------")
print("2 LAYERS")
for hidden_unit in hidden_units_2_layer:
  batch_size = 32
  layers = [AffineLayer(batch_size, hidden_unit[0]), ReLU(), AffineLayer(hidden_unit[0], hidden_unit[1]), ReLU(), AffineLayer(hidden_unit[1], 10)]
  get_training_and_testing_accuracies_and_plot(batch_size=batch_size, layers=layers)

print("----------------------------------")
print("3 LAYERS")
for hidden_unit in hidden_units_3_layer:
  batch_size = 32
  layers = [AffineLayer(batch_size, hidden_unit[0]), ReLU(), AffineLayer(hidden_unit[0], hidden_unit[1]), ReLU(), AffineLayer(hidden_unit[1], hidden_unit[2]), ReLU(), AffineLayer(hidden_unit[2], 10)]
  get_training_and_testing_accuracies_and_plot(batch_size=batch_size, layers=layers)


# ==============================================================================
# Section 15: 6.b) Two or Three Layers
# ==============================================================================

# %% Cell 11
batch_size=500
def get_layers(new_dimensions, batch_size=batch_size, include_drop_outs:float=0.):
  layers = [AffineLayer(batch_size, np.random.randint(batch_size)), ReLU()]

  index = -3 if include_drop_outs else -2
  if include_drop_outs:
    layers += [Dropout(include_drop_outs)]

  for new_dimension in new_dimensions:
    prev_layer = layers[index]
    prev_dimension = prev_layer.parameters[1].shape[0]
    current_layer_set = [
        AffineLayer(prev_dimension, new_dimension),
        ReLU()
    ]

    if include_drop_outs:
      current_layer_set += [Dropout(include_drop_outs)]

    layers += current_layer_set

  layers += [AffineLayer(layers[index].parameters[1].shape[0], 10)]
  return layers

# %% Cell 12
dimensions_to_test = [(128, 128), (128, 128, 128), (128, 128, 128, 128)]

for dimensions in dimensions_to_test:
  layers_to_test = get_layers(new_dimensions=dimensions)
  get_training_and_testing_accuracies_and_plot(layers=layers_to_test, batch_size=batch_size)

# ==============================================================================
# Section 16: 6.c) Number of Layers
# ==============================================================================

# %% Cell 13
dimensions_to_test = [(128, 128), (128, 128, 128), (128, 128, 128, 128)]


for dimensions in dimensions_to_test:
  layers_to_test = get_layers(new_dimensions=dimensions, include_drop_outs=0.5)
  get_training_and_testing_accuracies_and_plot(layers=layers_to_test, batch_size=batch_size)

# ==============================================================================
# Section 17: 6.d) Dropout
# ==============================================================================

# 6.e) Performance Impact  As the number of layers increase, training takes longer.
