# Extracted from linear-regression.ipynb
# NOTE: Code cells preserved; markdown condensed into section headers/comments

# ==============================================================================
# Section 1: Notebook
# ==============================================================================

# %% Cell 0
"""
  @author: Urmzd Mukhamamdnaim
  @description Solution for Q1 of CSCI4151's A0.
  @requires Python3.9
"""
import numpy as np
from typing import Tuple, Literal

# %% Cell 1
# NOTE: DELETE THE SEED DURING TESTING. 
np.random.seed(42)
# NOTE: POSSIBLY DELETE THIS LINE IF YOUR NOTEBOOK DOESN'T SUPPORT WIDGETS
# %matplotlib widget

# %% Cell 2
from matplotlib import pyplot as plt

RANGE = Tuple[int, int]

def chooseParams(range:RANGE=(-2,2)) -> np.ndarray:
  """
    @param [optional] range - The inclusive start and end of the uniform space to pick from

    @return - A numpy array of size (2,) where index 0 represents the true weight, and index 1 represents the true bias.

    @example
    // Returns np.ndarray([-0.1, 1.2])
    chooseParams((-2,2))
  """
  return np.random.uniform(range[0], range[1], size=(2,))

NOISE_TYPE = Literal["gaussian", "uniform"]
DATA_SET = Tuple[np.ndarray, np.ndarray]

def generateNoisyLinearData(n: float, xLo: float, xHi: float, w: float, b: float, noiseType: NOISE_TYPE, sigma: float, uniform_range:RANGE=(-1,1)) -> DATA_SET:
  """
    @param n - The number of samples to generate.
    @param xLo - The starting domain value.
    @param xHi - The ending domain value.
    @param w - The linear function's scale factor.
    @param b - The additional translation to apply.
    @param noiseType - The type of noise to include in the function.
    @param [sigma] - If "gaussian" noiseType is selected, this parameter specifies the desired standard deviation.

    @return - A tuple containing the generated X and Y values.
  """

  # Generate a collection of `n` samples in the range [xLo, xHi]
  x = np.random.uniform(xLo, xHi, size=(n,))
  mu = 0
  noise = np.random.uniform(*uniform_range, size=(n,)) if noiseType == "uniform" else np.random.normal(mu, sigma, size=(n,))
  y = w * x + b + noise
  return x,y

def plotLinearData(x: np.ndarray, y: np.ndarray, noiseType:NOISE_TYPE="uniform") -> None:
  """
    @param x - The domain values to plot.
    @param y - The range values to plot.
    @param [noiseType] - The type of noise used to generate the dataset.
  """
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.set_title(f"Linear Dataset Using {noiseType.title()} Distribution as Noise.")
  ax.set_xlabel("X")
  ax.set_ylabel("Y")

  plt.show()

xLo, xHi = (-3., 3.)
sigma = 3
n = 1000
w_gen, b_gen = chooseParams()
plotLinearData(*generateNoisyLinearData(n, xLo, xHi, w_gen, b_gen, "gaussian", sigma), "gaussian")
plotLinearData(*generateNoisyLinearData(n, xLo, xHi, w_gen, b_gen, "uniform", sigma), "uniform")

# %% Cell 3
"""
  @author Urmzd Mukhammadnaim
  @description Solution for Q2 of CSCI4155's A0.
  @requires Python3.9
"""

import numpy as np
from matplotlib import pyplot as plt
from typing import Literal

LOSS_TYPE = Literal["absolute", "squared"]

def loss(w: float, b: float, x: np.ndarray, ytrue: np.ndarray, lossType: LOSS_TYPE) -> np.ndarray:
  """
    @param w - The weight of the predicted Y value.
    @parma b - The bias of the preidcted Y value.
    @param x - The X value from which to predict the Y value from.
    @param ytrue - The true Y value.
    @pram lossType - The loss function to apply.

    @return - A numpy array of size (len(x),) containing the loss function for each associated `x` value.

    @example
    // Returns [2]
    loss(2, 1, [3], 9, "absolute")
  """
  y_pred = w * x + b
  loss = y_pred - ytrue
  return np.abs(loss) if lossType == "absolute" else np.square(loss)

def cost(w: float, b: float, dataset: DATA_SET, lossType: LOSS_TYPE) -> np.floating:
  """
    @param w - The estimated weight for the predicted Y vector.
    @param b - The estimated bias for the predicted Y vector.
    @param dataset - A tuple of (X, Y) values.
    @param lossType - The loss function to apply to each value in the X vector.

    @return - The average loss of every value associated with the dataset.
  """
  x, y = dataset;

  losses = loss(w, b, x, y, lossType)
  cost = np.average(losses)
  return cost / 2

def plot_cost(b: float, dataset: DATA_SET, lossType: LOSS_TYPE, noiseType: NOISE_TYPE = "uniform") -> None:
  """
    @param b - The bias to use when predicting the Y values.
    @param dataset - The dataset to predict and compare Y values from.
    @param lossType - The loss function to use when calculating the error.
    @param noiseType - The type of noise that has been applied to the dataset.

    @return - Displays a plot demonstrating the relationship between cost as a dependency of weight.
  """
  _, ax = plt.subplots()

  w_space = np.linspace(-2, 2, 100, True)
  costs = np.array([cost(w, b, dataset, lossType) for w in w_space])

  ax.scatter(w_space, costs)
  ax.set_title(f"Cost as a Product of Weight ({lossType.title()} Loss Function, {noiseType.title()} Distribution)")
  ax.set_xlabel(f"Weight")
  ax.set_ylabel("Cost")
  plt.show()

xLo, xHi = (-3, 3)
n = 100
sigma = 3
w, b = chooseParams()
gaussian_dataset = generateNoisyLinearData(n, xLo, xHi,w, b, "gaussian", sigma)
uniform_dataset = generateNoisyLinearData(n, xLo, xHi, w, b, "uniform", sigma)
plot_cost(4, gaussian_dataset, "absolute", "gaussian")
plot_cost(4, gaussian_dataset, "square", "gaussian")
plot_cost(4, uniform_dataset, "absolute")
plot_cost(4, uniform_dataset, "square")

# ==============================================================================
# Section 2: Prefix
# ==============================================================================

# %% Cell 4
"""
  @author Urmzd Mukhammadnaim
  @description Solution of Q3a for CSCI 4155's A0.
"""
import numpy as np
from matplotlib import pyplot as plt


def plot_3d_graph(dataset: DATA_SET, show_contour=False, lossType: LOSS_TYPE = "square") -> None:
  W = np.linspace(-5, 5, 1000)
  B = np.linspace(-5, 5, 1000)

  C = np.array([cost(w, b, dataset, lossType) for w in W for b in B])

  W, B = np.meshgrid(W, B)
  C = C.reshape(W.shape)

  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  if show_contour:
    ax.remove()
    ax = fig.add_subplot()
    ax.contourf(W, B, C)
  else:
    ax.plot_surface(W, B, C)
  ax.set_xlabel("Weight")
  ax.set_ylabel("Bias")
  if not show_contour:
    ax.set_zlabel("Cost")
    ax.set_title("Cost as a product of Bias and Weight")


  plt.show()

# %% Cell 5
xLo, xHi = (-3., 3.)
sigma = 3
n = 1000
w_gen, b_gen = chooseParams()
dataset = generateNoisyLinearData(n, xLo, xHi, w_gen, b_gen, "gaussian", sigma)
plot_3d_graph(dataset)

# ==============================================================================
# Section 3: Question 3b)
# ==============================================================================

# %% Cell 6
plot_3d_graph(dataset, show_contour=True)

# ==============================================================================
# Section 4: Question 3c)
# ==============================================================================

# Given the plots above, the lowest cost appears around w ~= 1.5 and b ~= -0.2.

# ==============================================================================
# Section 5: Question 4
# ==============================================================================

# Cost (scalar):
#   C = (1 / (2N)) * sum_i (y_pred_i - y_i)^2
#
# Vector form (per-sample):
#   C = (1 / (2N)) * (w_p * x_i + b_p - y_i)^2

# ==============================================================================
# Section 6: 4a)
# ==============================================================================

# dC/db_p = (-b_p + y_i) / N
# Setting derivative to 0 yields b_p = y_i (sanity check: b_p == b_true).

# ==============================================================================
# Section 7: 4b)
# ==============================================================================

# dC/dw_p = (w_p * x_i - y_i) * w_p / N
# Setting derivative to 0 yields w_p = y_i / x_i (sanity check: w_p == w_true).

# ==============================================================================
# Section 8: Question 4c)
# ==============================================================================

# Solve for w_p and b_p:
#   dC/dw_p = (w_p x_i + b_p - y_i) * x_i / N
#   dC/db_p = (w_p x_i + b_p - y_i) / N
# => w_p x_i + b_p = y_i

# ==============================================================================
# Section 9: Question 5
# ==============================================================================

# No written response was provided in the original notebook for this heading.

# ==============================================================================
# Section 10: Question 5a)
# ==============================================================================

# %% Cell 7
"""
  @author Urmzd Mukhammadnaim
  @description Solution for question 5b of CSCI's 4155 A0.
"""
def computeDeriv_dC_db(b: float, w: float, X: np.ndarray, Y: np.ndarray) -> float:
  """
    Derive the cost with respect to the bias.
    @param b - The bias value to caluclate the cost with.
    @param w - The predicted weight.
    @param X - X values to transform.
    @param Y - True Y's.
  """
  derivative = np.average((w*X + b - Y))
  return derivative 

def computeDeriv_dC_dw(b: float, w: float, X: np.ndarray, Y: np.ndarray) -> float:
  """
    Derive the cost with respect to the weight.
    @param b - Bias associated with cost.
    @param w - The weight value to calculate the cost with. 
    @param X - X values to transform.
    @param Y - True Y's.
  """
  derivative = np.average(((w * X + b - Y) * X))
  return derivative 

# ==============================================================================
# Section 11: Question 5c)
# ==============================================================================

# %% Cell 8
"""
  @author - Urmzd Mukhammadnaim
  @description - Solution for question 6 of CSCI 4155.
"""
WEIGHT_BIAS_PAIR = Tuple[np.ndarray, np.ndarray, np.ndarray]
def updateParams(w: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, alpha: float, epochs: int) -> WEIGHT_BIAS_PAIR:
  """
    @param w - Starting weight.
    @param b - Starting bias.
    @param X - True X's.
    @param Y - True Y's.
    @param alpha - learning rate.
    @param epoch - number of iterations to update and modify gradient.
    
    @returns - (W, B, E)
  """
  w_vec = np.zeros(shape=(epochs+1,))
  b_vec = np.zeros(shape=(epochs+1,))
  error_vec = np.zeros(shape=(epochs+1,))
  w_vec[0] = w;
  b_vec[0] = b;
  error_vec[0] = cost(w, b, (X, Y), "square")

  for i in range(1, epochs+1):
    w_deriv = computeDeriv_dC_dw(b, w, X, Y)
    b_deriv = computeDeriv_dC_db(b, w, X, Y)
    w = w_vec[i] = w - alpha * w_deriv 
    b = b_vec[i] = b - alpha * b_deriv
    error_vec[i] = cost(w, b, (X, Y), "square")

  return w_vec, b_vec, error_vec 

# %% Cell 9
# %matplotlib inline
"""
  @author - Urmzd Mukhammadnaim
  @description - Question 6B for CSCI 4155
"""
w, b = chooseParams()
X, Y = generateNoisyLinearData(100, -10, 10, w, b, "gaussian", 2.3)
w_random, b_random = chooseParams()
alpha = 0.01
W, B, E = updateParams(w_random, b_random, X, Y, alpha, 10000)
print(W[-1], w, B[-1], b, E[-1], "TRUEW PRED TRUEB PRED E")

# %% Cell 10
def plot_cost_wrt_epochs(error_vec: np.ndarray = E, alpha: float=alpha) -> None:
  _, ax = plt.subplots()

  ax.set_title(f"Cost with respect to # of Epochs, LR = {alpha}")
  ax.set_xlabel("# of Epochs")
  ax.set_ylabel("Cost")
  ax.grid(True, "both", "both")
  ax.plot(error_vec)

# %% Cell 11
plot_cost_wrt_epochs()
_, _, E1 = updateParams(w_random, b_random, X, Y, 0.1, 10000)
_, _, E2 = updateParams(w_random, b_random, X, Y, 1, 10000)
_, _, E3 = updateParams(w_random, b_random, X, Y, 10, 10000)
_, _, E4 = updateParams(w_random, b_random, X, Y, 100, 10000)
plot_cost_wrt_epochs(E1, 0.1)
plot_cost_wrt_epochs(E2, 1)
plot_cost_wrt_epochs(E3, 10)
plot_cost_wrt_epochs(E4, 100)

# %% Cell 12
def plot_contour_plot_with_parameter_markers(W=W, B=B) -> None:
  _, ax = plt.subplots(figsize=(10, 10))

  ax.tricontour(W, B, E, 100, alpha=0.5, cmap="BuPu")
  ax.plot(W[0], B[0], 'rx', zorder=1, label="Starting W, B", linewidth=2, markersize=10)
  ax.plot(W[1:-1], B[1:-1], 'c.', zorder=1, label="Intermediate W, B", linewidth=2, markersize=10)
  ax.plot(W[-1], B[-1], 'bo', zorder=1, label="Ending W, B", linewidth=2, markersize=10)
  ax.set_label("Weight")
  ax.set_label("Bias")
  ax.set_title("Cost as a product of Weight and Bias")
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# %% Cell 13
plot_contour_plot_with_parameter_markers()

# Question 6B  With higher learning rates, the cost might reach near optimum values quickly but as the number of epochs increase, will diverge (as seen in the later cost-epoch charts).  With lower learning rates, the cost function will take a longer period of time to hit near-optimum values, having a asymptomic-like behaviour as epochs continue.

# %% Cell 14
"""
  @author Urmzd Mukhammadnaim
  @description Answer for Question #7 of CSCI 4155 A0. 
"""
def regularizer(w: np.floating, b: np.floating, Lambda: float) -> np.floating: 
  """
    @param L - The parameter to penalize.
    @param w - The updated weight.
    @param b - The updated bias.
    @param Lambda - The regularization factor.
  """
  return (Lambda/2  * (np.power(w, 2) + np.power(b, 2)))

WEIGHT_BIAS_ERROR_PAIR = Tuple[np.ndarray, np.ndarray, np.ndarray]
def updateParamsWithReg(w: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, alpha: float, epochs: int, Lambda: float) -> WEIGHT_BIAS_ERROR_PAIR:
  """
    @param w - Starting weight.
    @param b - Starting bias.
    @param X - True X's.
    @param Y - True Y's.
    @param alpha - learning rate.
    @param epoch - number of iterations to update and modify gradient.
    @param Lambda - the penalizing/regularization factor.
    
    @returns - (W, B, E) -> Weight, Bias, Error
  """
  w_vec = np.zeros(shape=(epochs+1,))
  b_vec = np.zeros(shape=(epochs+1,))
  error_vec = np.zeros(shape=(epochs+1,))

  w_vec[0] = w;
  b_vec[0] = b;
  reg_fac = regularizer(w, b, Lambda)
  error_vec[0] = cost(w, b, (X, Y), "square") + reg_fac

  for i in range(1, epochs+1):
    reg_fac = regularizer(w, b, Lambda)
    w_deriv = computeDeriv_dC_dw(b, w, X, Y)
    b_deriv = computeDeriv_dC_db(b, w, X, Y)
    w = w - alpha * w_deriv + reg_fac 
    b = w - alpha * b_deriv + reg_fac 
    w_vec[i] = w 
    b_vec[i] = b 
    error_vec[i] = cost(w, b, (X, Y), "square")

  return w_vec, b_vec, error_vec 

# %% Cell 15
W6, B6, E6 = updateParamsWithReg(w, b, X, Y, 0.01, 10000, 0.01)
print(w, W6[-1], b, B6[-1], E6[-1], "WTRUE, WPRED, BTRUE, BPRED, COST")

# ==============================================================================
# Section 12: Question 8
# ==============================================================================

# Author: Urmzd Mukhammadnaim
# 
# Description: Solution for CSCI 4155 A0 Q8
# Requires: Python 3.10+

# ==============================================================================
# Section 13: Question 8.a)
# ==============================================================================

# %% Cell 16
import numpy as np
import matplotlib.pyplot as plt

x, y = np.load("dataset_cos.npy")
print(x, y)

# Verify function is child of Cos function.
_, ax = plt.subplots()
ax.scatter(x, y)

# ==============================================================================
# Section 14: Question 8.b)
# ==============================================================================

# %% Cell 17
print(x.size, y.size)
W, B, E = updateParams(0, 0, x, y, 0.01, 5000)
print(E.size)
print(W[-1], B[-1])
print(E[0], E[-1])

_, ax = plt.subplots()
ax.scatter(list(range(E.size)), E)

# $$ \begin{aligned} \mathcal{C} &= \frac{1}{N} \cdot \sum{i=1}^{N}{(0 - \cos{(2x + 1)})^2}\\ &= \frac{1}{N} \cdot \sum{i=1}^{N}{\cos^2{(2x + 1)}} \end{aligned} $$

# %% Cell 18
analytical_cost = np.average(x**2) 
print (f"Analytical Cost: {analytical_cost}, Gradient Descent COST: {E[-1]}")

# Conclusion  The analytical cost is approximately $14.26$ whereas the cost derived using gradient descent is approximately $0.22$. The two values differ by a little under $70$ magnitudes.

# ==============================================================================
# Section 15: Question 8.c)
# ==============================================================================

# %% Cell 19
def plot_cost_wrt_epochs(error_vec: np.ndarray, alpha: float, w: float) -> None:
  _, ax = plt.subplots(figsize=(10,10))

  ax.set_title(f"Cost with respect to # of Epochs, LR = {alpha}, W_init={w}")
  ax.set_xlabel("# of Epochs")
  ax.set_ylabel("Cost")
  ax.grid(True, "both", "both")
  ax.plot(error_vec)

def computeDeriv_dC_dw(b: float, w: float, X: np.ndarray, Y: np.ndarray) -> float:
  a = -2 * X * (np.sin(w * X + b))
  b = np.cos(w * X + b) - Y
  N = X.shape[0]
  return a.dot(b) / N

def computeDeriv_dC_db(b, w, X, Y) -> float:
  a = -2 * np.sin(w * X + b)
  b = np.cos(w * X + b) - Y
  N = X.shape[0]
  return a.dot(b) / N

def loss(w: float, b: float, x: np.ndarray, ytrue: np.ndarray, lossType: LOSS_TYPE) -> np.ndarray:
  y_pred = np.cos(w * x + b)
  loss = y_pred - ytrue
  return np.abs(loss) if lossType == "absolute" else np.square(loss)

def cost(w: float, b: float, dataset: DATA_SET, lossType: LOSS_TYPE) -> np.floating:
  x, y = dataset;

  losses = loss(w, b, x, y, lossType)
  cost = np.average(losses)
  return cost

def updateParams(w: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, alpha: float, epochs: int) -> WEIGHT_BIAS_PAIR:
  w_vec = np.zeros(shape=(epochs+1,))
  b_vec = np.zeros(shape=(epochs+1,))
  error_vec = np.zeros(shape=(epochs+1,))

  w_vec[0] = w;
  b_vec[0] = b;
  error_vec[0] = cost(w, b, (X, Y), "square")

  for i in range(1, epochs+1):
    w_deriv = computeDeriv_dC_dw(b, w, X, Y)
    b_deriv = computeDeriv_dC_db(b, w, X, Y)
    w = w_vec[i] = w - alpha * w_deriv 
    b = b_vec[i] = b - alpha * b_deriv
    error_vec[i] = cost(w, b, (X, Y), "square")

  return w_vec, b_vec, error_vec 

# ==============================================================================
# Section 16: Question 8.d)
# ==============================================================================

# %% Cell 20
for w in [0.1, 0.5, 1.5, 1.9]:
  alpha = 0.05
  epochs = 1000
  W, B, E = updateParams(w, 0, x, y, alpha, epochs)
  plot_cost_wrt_epochs(E, alpha, w)

# Conclusion  As seen below, the cost function has a trignometric base. Due to this, there are multiple local extremas. As seen above, certain weights near these extremums have the possibility of approaching a non global optimum, resulting in costs having asymptomic like behaviour (plateauing) far above the desired cost, $0$, as seen in examples where $w<1.9$.

# %% Cell 21
_, axs = plt.subplots(2, 1, figsize=(10, 10))

w = np.linspace(-3, 3, 200)
b = np.linspace(-3, 3, 200)

cs = [
        np.array([cost(2, b[i], (x, y), "square") for i in range(len(b))]),
        np.array([cost(w[i], 1, (x, y), "square") for i in range(len(w))])
]

titles = [
        "Bias with respect to Cost",
        "Weight with respect to Cost"
]

parameter = [
        b, 
        w
]

for i, ax in enumerate(axs):
        print(i, len(parameter), len(titles), len(cs))
        ax.plot(parameter[i], cs[i])
        ax.set_title(titles[i])
        ax.grid()

# ==============================================================================
# Section 17: Question 8.e)
# ==============================================================================

# %% Cell 22
def updateParams(w: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, alpha: float, epochs: int, mu: float = 0.99) -> WEIGHT_BIAS_PAIR:
  m_w_vec = np.zeros(shape=(epochs + 1))
  m_b_vec = np.zeros(shape=(epochs + 1))

  w_vec = np.zeros(shape=(epochs+1,))
  b_vec = np.zeros(shape=(epochs+1,))
  error_vec = np.zeros(shape=(epochs+1,))

  m_w = 0
  m_b = 0

  m_w_vec[0] = m_w;
  m_b_vec[0] = m_b;

  w_vec[0] = w;
  b_vec[0] = b;

  error_vec[0] = cost(w, b, (X, Y), "square")

  for i in range(1, epochs+1):
    w_deriv = computeDeriv_dC_dw(b, w, X, Y)
    b_deriv = computeDeriv_dC_db(b, w, X, Y)

    m_w = m_w_vec[i] = mu * m_w_vec[i - 1] + w_deriv
    m_b = m_b_vec[i] = mu * m_b_vec[i - 1] + b_deriv

    w = w_vec[i] = w - alpha * m_w
    b = b_vec[i] = b - alpha * m_b

    error_vec[i] = cost(w, b, (X, Y), "square")

  return w_vec, b_vec, error_vec 

# %% Cell 23
for w in [0.1, 0.5, 1.5, 1.9]:
  alpha = 0.05
  epochs = 1000
  W, B, E = updateParams(w, 0, x, y, alpha, epochs)
  plot_cost_wrt_epochs(E, alpha, w)

# Conclusion  More instances converge with momentum than without it. The reason for this lies in the momentum function given. More precisely, the reason lies in how smaller extremas are passed in favour of a larger (potentially global) extrema.  To build a simple understanding of what was said, imagine that you're at the at the top of a very large hill. At this point, you have alot of potential energy stored up. Once you began your descent via a skateboard for example, the potential energy turns into kinetic energy, and in the further you go down, the larger your momentum becomes. Now, what happens when you reach a small dip (convex)? Given that your momentum is high enough, you pass over the dip and continue riding until you reach a dip with a significantly large slope. At this point, you simple just oscillate until a stop condition is met.  Now this explaination is pretty simple, but it should demonstrate what I'm attempting to state.  To understand why, lets take a look at the momentum equation  $$ m{t+1, p} = \mu \cdot m{t,p} + \frac{\partial{L}}{\partial{P}} $$  aswell as the parameter update equation.  $$ p{t+1} = p {t} - \alpha \cdot m{t+1, p} $$  In the momentum equation, we see that the previous momentum is used in the calculation of the current momentum. The momentum will continue increasing given that the parameter changes in the same direction, if it doesn't, the momentum decays. If momentum is great enough, certain parameter values will jump signifcantly in order to reach a potentially "better" extrema (as demonstrated by the oscillation in the above graphs).

# ==============================================================================
# Section 18: Question 8.f)
# ==============================================================================

# %% Cell 24
def updateParams(w: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, alpha: float, epochs: int, mu: float = 0.99) -> WEIGHT_BIAS_PAIR:
  w_vec = np.zeros(shape=(epochs+1,))
  b_vec = np.zeros(shape=(epochs+1,))
  error_vec = np.zeros(shape=(epochs+1,))

  w_vec[0] = w;
  b_vec[0] = b;

  error_vec[0] = cost(w, b, (X, Y), "square")

  for i in range(1, epochs+1):
    index = np.random.choice(len(X), size=(1,))
    X_sample = X[index]
    Y_sample = Y[index]
    w_deriv = computeDeriv_dC_dw(b, w, X_sample, Y_sample)
    b_deriv = computeDeriv_dC_db(b, w, X_sample, Y_sample)

    w = w_vec[i] = w - alpha * w_deriv 
    b = b_vec[i] = b - alpha * b_deriv 

    error_vec[i] = cost(w, b, (X_sample, Y_sample), "square")

  return w_vec, b_vec, error_vec 

# %% Cell 25
for w in [0.1, 0.5, 1.5, 1.9]:
  alpha = 0.05
  epochs = 1000
  W, B, E = updateParams(w, 0, x, y, alpha, epochs)
  plot_cost_wrt_epochs(E, alpha, w)
  print(f"FINAL PARAMETERS W={W[-1]} B={B[-1]} C={E[-1]}")

# Conclusion  The method does improve convergence.

# ==============================================================================
# Section 19: Question 8.g)
# ==============================================================================

# %% Cell 26
def updateParams(w: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray, alpha: float, epochs: int, mu: float = 0.99) -> WEIGHT_BIAS_PAIR:
  m_w_vec = np.zeros(shape=(epochs + 1))
  m_b_vec = np.zeros(shape=(epochs + 1))

  w_vec = np.zeros(shape=(epochs+1,))
  b_vec = np.zeros(shape=(epochs+1,))
  error_vec = np.zeros(shape=(epochs+1,))

  m_w = 0
  m_b = 0

  m_w_vec[0] = m_w;
  m_b_vec[0] = m_b;

  w_vec[0] = w;
  b_vec[0] = b;

  error_vec[0] = cost(w, b, (X, Y), "square")

  for i in range(1, epochs+1):
    index = np.random.choice(len(X), size=(1,))
    X_sample = X[index]
    Y_sample = Y[index]

    w_deriv = computeDeriv_dC_dw(b, w, X_sample, Y_sample)
    b_deriv = computeDeriv_dC_db(b, w, X_sample, Y_sample)

    m_w = m_w_vec[i] = mu * m_w_vec[i - 1] + w_deriv
    m_b = m_b_vec[i] = mu * m_b_vec[i - 1] + b_deriv

    w = w_vec[i] = w - alpha * m_w
    b = b_vec[i] = b - alpha * m_b

    error_vec[i] = cost(w, b, (X_sample, Y_sample), "square")

  return w_vec, b_vec, error_vec 

# %% Cell 27
for w in [0.1, 0.5, 1.5, 1.9]:
  alpha = 0.05
  epochs = 1000
  W, B, E = updateParams(w, 0, x, y, alpha, epochs)
  plot_cost_wrt_epochs(E, alpha, w)
  print(f"FINAL PARAMETERS W={W[-1]} B={B[-1]} C={E[-1]}")

# %% Cell 28
for w in [0.1, 0.5, 1.5, 1.9]:
  alpha = 0.0005
  epochs = 1000
  W, B, E = updateParams(w, 0, x, y, alpha, epochs)
  plot_cost_wrt_epochs(E, alpha, w)
  print(f"FINAL PARAMETERS W={W[-1]} B={B[-1]} C={E[-1]}")

# Conclusion  When using momentum with stochastic gradient descent with the same learning rate as before, we find that while the cost remains relatively low, the weight and bias are signicantly different (multiple orders of magnitude different). This is likely during the periodicity associated with trigonometric function. In other words, the algorithm looks significantly beyond its immediate surroundings and jumps from parameter value to parameter value in a noteworthy way.  This is not the case as the learning rate decrease to $0.0005$. In the beginning they're is a lot of sporadicity, however, this dies down as the number of epochs increase.
