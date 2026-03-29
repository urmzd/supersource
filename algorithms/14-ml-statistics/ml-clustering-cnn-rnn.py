# Extracted from A4.ipynb
# NOTE: Code cells preserved; markdown condensed into section headers/comments

# ==============================================================================
# Section 1: Notebook
# ==============================================================================

# CSCI3151 - Foundations of Machine Learning
# Assignment 4 (Summer 2021)

# Your assignment is to be submitted as a single .ipynb file (please do not zip it when submitting to brightspace) including your answers to both the math and the experimental questions, in the correct order, on Brightspace. Use [markdown syntax](https://www.markdownguide.org/cheat-sheet/) to format your answers

# Note: in solving the math questions, aim for general (symbolic) solutions and substitute the specific numbers at the end. This demonstrates a solid understanding of the key concepts. You can answer the math questions in two ways:  Use LaTeX to typeset the equations. Section H of [this LaTeX reference sheet](http://tug.ctan.org/info/latex-refsheet/LaTeXRefSheet.pdf) is a good reference. Here is another [LaTeX reference sheet](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference). The equations in the questions are typeset in LaTeX, so you can use them as examples.  Use neat handwriting, scan your solution using [Camscanner](https://www.camscanner.com/user/download) on your mobile phone, upload the image file, and embed it in your solution notebook. To this end (1) create an empty Markdown cell. 2) Drag-and-drop the image file into the empty Markdown cell, or click on the image icon at the top of the cell and select the image file. The Markdown code that will embed the image, together with its content, then appears.

# Your answers to the experimental questions should be in your solution notebook, in the form of code and text cells, using markdown for your text responses. You should also include the results of running your code.

# The marking criteria are described in rubrics.

# You can submit multiple editions of your assignment. Only the last one will be marked. It is recommended to upload a complete submission, even if you are still improving it, so that you have something into the system if your computer fails for whatever reason.  IMPORTANT: PLEASE NAME YOUR PYTHON NOTEBOOK FILE AS:  <LASTNAME-<FIRSTNAME-Assignment-N.ipynb  for example: Soto-Axel-Assignment-4.ipynb \\

# ==============================================================================
# Section 2: 1. Clustering
# ==============================================================================

# In this question we are going to explore two different clustering methods on the [Wine Dataset](https://archive.ics.uci.edu/ml/datasets/Wine) and evaluate it using two measures: one is an intrinsic measure (no labels), while the other one makes use of the available labels.

# %% Cell 0
from sklearn.datasets import load_wine
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler


def load_dataset():
    scalar = StandardScaler()
    wine_dataset = load_wine()

    X = scalar.fit_transform(wine_dataset.data)
    y = wine_dataset.target

    return X, y


# a) Cluster the dataset using the [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), and [k-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) clustering algorithm without using the class information as part of the features. Experiment with different numbers of clusters ranging from 2 to 5.

# %% Cell 1
from sklearn.metrics import silhouette_score, adjusted_rand_score


def cluster_metrics(cls, X, y, n_clusters=2):
    pred = cls(n_clusters=n_clusters).fit_predict(X)
    silhouette_cof = silhouette_score(X, pred)
    adj_rand_index = adjusted_rand_score(y, pred)
    return n_clusters, silhouette_cof, adj_rand_index


# b) What is the variability of the resulting clusters as a function of different initializations or parameterization? Use the Silhouette coefficient and Adjusted Rand Index as metrics for evaluation to discuss the stability of results.

# %% Cell 2
X, y = load_dataset()

for cls in (AgglomerativeClustering, KMeans):
    for i in range(2, 6):
        cluster = cluster_metrics(cls, X, y, n_clusters=i)
        print(f"-- {cls.__name__}, No. of Clusters = {cluster[0]} --")
        print(f"-- Silhoutte Coefficient -- ")
        print(f"{cluster[1]}")
        print(f"-- Adjusted Rand Index --")
        print(f"{cluster[2]}")
        print()

# ==============================================================================
# Section 3: Summary
# ==============================================================================

# As the number of clusters approaches 3, the model becomes optimal. The adjusted
# Rand index (~0.79) suggests strong agreement with labels, while the silhouette
# coefficient (~0.27) indicates overlap between clusters. Overall, 3 clusters
# fits the data best.

# c) Based on the Silhouette coefficient, discuss (i) which clustering method you would pick, (ii) how many clusters you would use for your data.  Make sure that appropriate visualizations are used to support the analysis.

# %% Cell 3
import matplotlib.pyplot as plt

X, y = load_dataset()

fig, ax = plt.subplots()
X_range = list(range(2, 6))
for cls in (AgglomerativeClustering, KMeans):
    data = [cluster_metrics(cls, X, y, n_clusters=i) for i in X_range]
    sils = [d[1] for d in data]
    rand = [d[2] for d in data]

    ax.plot(X_range, sils, label=f"{cls.__name__} Silhoutte Coefficient")
    ax.plot(X_range, rand, label=f"{cls.__name__} Adjusted Rand Index")


ax.legend(bbox_to_anchor=(1.05, 1))

# ==============================================================================
# Section 4: Analysis
# ==============================================================================

# (i) Prefer K-Means based on the silhouette coefficient.
# (ii) The metrics are optimal around N=3; for N > 3 they deviate, suggesting
#     worse clustering quality.

# ==============================================================================
# Section 5: 2. Convolutional neural networks
# ==============================================================================

# In this question you will construct a convolutional neural network to classify a large set of low resolution images. Similarly to what you have done with A2, we would like you to describe the behavior of the network as you modify certain parameters.  Use the fashionmnist dataset (available from Keras):

# #####Understanding the dataset

# %% Cell 4
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(x_train_original, y_train_original), (x_test_original, y_test_original) = (
    fashion_mnist.load_data()
)

# Normalize data.
x_train = x_train_original / 255.0
x_test = x_test_original / 255.0

# %% Cell 5
print("Training data shape: ", x_train_original.shape)
print("Training labels shape: ", y_train_original.shape)
print("Test data shape: ", x_test_original.shape)
print("Test labels shape: ", y_test_original.shape)

# %% Cell 6
import matplotlib.pyplot as plt

imgplot = plt.imshow(x_test[64])
plt.show()

# ###a) Using two convolutional layers explore the impact on different choices for the number of nodes and filter sizes for the two layers. Summarize your observations.

# %% Cell 7
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, InputLayer
from keras.optimizers import Adam
import numpy as np
from tensorflow.keras.utils import to_categorical


def dataset_init():
    (x_train_original, y_train_original), (x_test_original, y_test_original) = (
        fashion_mnist.load_data()
    )

    Y_train = to_categorical(y_train_original)
    Y_test = to_categorical(y_test_original)

    X_train_norm = x_train_original.astype("float32") / 255.0
    X_test_norm = x_test_original.astype("float32") / 255.0

    return X_train_norm, Y_train, X_test_norm, Y_test


def build_network(n_nodes, filter_size):
    cnn = Sequential()
    print(n_nodes, filter_size)
    assert len(n_nodes) == len(filter_size), (
        "No of nodes and filter size shape do not match."
    )

    cnn.add(
        Conv2D(n_nodes[0], filter_size[0], activation="relu", input_shape=(28, 28, 1))
    )
    cnn.add(Conv2D(n_nodes[1], filter_size[1], activation="relu"))

    cnn.add(Flatten())
    cnn.add(Dense(10, activation="softmax"))

    cnn.compile(
        optimizer="adam",
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"],
    )

    return cnn


def train_network(network: Sequential, X, y, X_test, y_test, epochs=10):
    history = network.fit(
        np.expand_dims(X, -1),
        y,
        batch_size=1024,
        epochs=epochs,
        validation_data=(np.expand_dims(X_test, -1), y_test),
    )
    return history.history


def plot_network(
    x_range, n_nodes, filter_size, epochs, X, y, X_test, y_test, title="Varying ..."
):
    fig, ax = plt.subplots()

    assert len(n_nodes) == len(filter_size) == len(epochs) == len(x_range), (
        "Shape of inputs not same."
    )

    networks = [build_network(n_nodes[i], filter_size[i]) for i in range(len(x_range))]
    histories = [
        train_network(networks[i], X, y, X_test, y_test, epochs=epochs[i])
        for i in range(len(x_range))
    ]

    losses = [np.mean(history["loss"]) for history in histories]
    accuracies = [np.mean(history["accuracy"]) for history in histories]
    val_losses = [np.mean(history["val_loss"]) for history in histories]
    val_accuracies = [np.mean(history["val_accuracy"]) for history in histories]

    _range = [str(_x_range) for _x_range in x_range]
    print(losses, accuracies, val_losses, val_accuracies, range)

    assert len(_range) == len(losses), "Range != Losses"

    ax.plot(_range, losses, label="Training Loss")
    ax.plot(_range, accuracies, label="Training Accuracy")
    ax.plot(_range, val_losses, label="Testing Loss")
    ax.plot(_range, val_accuracies, label="Testing Accuracy")

    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title(f"{title}")


X_train, y_train, X_test, y_test = dataset_init()

n_nodes_default = ((8,) * 2,) * 2
filter_size_default = ((2,) * 2,) * 2
epochs_default = (4,) * 2

n_nodes_increasing = ((8,) * 2, (16,) * 2)
filter_size_increasing = ((2,) * 2, (3,) * 2)

print(len(n_nodes_default), n_nodes_default)
print(len(filter_size_default), filter_size_default)
print(len(epochs_default), epochs_default)

plot_network(
    n_nodes_increasing,
    n_nodes_increasing,
    filter_size_default,
    epochs_default,
    X_train,
    y_train,
    X_test,
    y_test,
    "Varying number of nodes.",
)
plot_network(
    filter_size_increasing,
    n_nodes_default,
    filter_size_increasing,
    epochs_default,
    X_train,
    y_train,
    X_test,
    y_test,
    "Varying number of filters.",
)

# ==============================================================================
# Section 6: Summary
# ==============================================================================

# Around 16 nodes per layer with filter size 3 appears to give the best balance
# of loss and accuracy for this CNN configuration.

# ###b) For the hyperparameters for the layers that you determined in part (a), experiment with a higher number of epochs. Summarize your observations.

# %% Cell 8
n_nodes_optimal = ((16,) * 2) * 8
filter_size_optimal = ((3,) * 2,) * 8
epochs_increasing = (4 * i for i in range(1, 9))
plot_network(
    filter_size_increasing,
    n_nodes_default,
    filter_size_increasing,
    epochs_default,
    X_train,
    y_train,
    X_test,
    y_test,
    "Varying number of Epochs.",
)

# ==============================================================================
# Section 7: Summary
# ==============================================================================

# Summary note: the original notebook did not include a written summary for the
# epoch-sweep experiment (section b).

# ==============================================================================
# Section 8: 3. Recurrent Neural Networks
# ==============================================================================

# In this question you will experiment with a simple recurrent neural network, where you will try to model a sinusoidal function with noise, whose amplitude becomes larger and larger as the independent variable t increases (0 ≤ t ≤ N). This function can be expressed in Python as:  x=(np.sin(0.02t)+2np.random.rand(N))(t/N).  For N=5000 we have:  [image omitted: see previous commit machine-learning/A4.ipynb]

# The idea is that you will train a recurrent neural network with points up to a certain value, Tp. This is all the training points will be t ≤ Tp . The length of the sequence provided to the network is a parameter that you can tune.

# a) Complete the provided code to fulfill this task. You need to add at least one simpleRNN layer and a proper output layer. Compare the models with different numbers of RNN cells. [8, 16, 32, 64]  Choose the best number of internal nodes for this model.

# %% Cell 9
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# sourced from this question description, transform data into matrix form
def putAsMatrix(data, length):
    X, Y = [], []
    for i in range(len(data) - length):
        d = i + length
        X.append(data[i:d,])
        Y.append(data[d,])
    return np.array(X), np.array(Y)


def data_init(length=4, show_plot=True):
    # get the data
    N = 5000
    Tp = 800

    t = np.arange(0, N)
    x = (np.sin(0.02 * t) + 2 * np.random.rand(N)) * (t / N)
    df = pd.DataFrame(x)
    print(df.head())

    if show_plot:
        plt.plot(df)
        plt.show()

    values = df.values
    train, test = values[0:Tp, :], values[Tp:N, :]

    # add length elements into train and test
    test = np.append(test, np.repeat(test[-1,], length))
    train = np.append(train, np.repeat(train[-1,], length))

    trainX, trainY = putAsMatrix(train, length)
    testX, testY = putAsMatrix(test, length)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    print(trainX.shape, trainY.shape)

    return trainX, trainY, testX, testY


# %% Cell 10
trainX, trainY, testX, testY = data_init()


# %% Cell 11
def simpleRNN(units, length=4):  # units is the number of recurrent nodes in your NN
    model = Sequential()
    # Include here the specification of your network
    # **********************************************
    # model.add(...)
    model.add(SimpleRNN(units))
    model.add(Dense(1))
    # ...
    # **********************************************
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.fit(trainX, trainY, validation_data=(testX, testY))
    model.summary()

    model.fit(trainX, trainY, epochs=50, batch_size=16)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    predicted = np.concatenate((trainPredict, testPredict), axis=0)

    scores = model.evaluate(testX, testY, verbose=0)
    print(scores)

    # the vertical red line shows the point where testing data starts
    index = df.index.values
    plt.plot(df)
    plt.plot(index, predicted)
    plt.axvline(df.index[Tp], c="r")
    plt.title(f"Simple RNN with {units} RNN units with length {length}")
    plt.show()


# %% Cell 12
simpleRNN(8)
simpleRNN(16)
simpleRNN(32)
simpleRNN(64)

# ==============================================================================
# Section 9: Summary
# ==============================================================================

# 64 RNN units performed best among the tested sizes.

# b) Starting with length = 4, discuss how different choices of the length of the sequence fed to the network can have an impact on performance.

# %% Cell 13
lengths = [4 * i for i in range(1, 9)]

for length in lengths:
    trainX, trainY, testX, testY = data_init(show_plot=False)
    simpleRNN(64, length)


# ==============================================================================
# Section 10: Summary
# ==============================================================================

# As sequence length increases with a fixed number of nodes, performance
# degrades, suggesting overfitting.
