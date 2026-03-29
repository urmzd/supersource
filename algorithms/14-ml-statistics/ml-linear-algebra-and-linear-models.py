# Extracted from A1.ipynb
# NOTE: Code cells preserved; markdown condensed into section headers/comments

# ==============================================================================
# Section 1: Notebook
# ==============================================================================

# CSCI3151 - Foundations of Machine Learning
# Assignment 1 (Summer 2021)
# Due: 7 June 2021, 23:30 ADT

# Submit your assignment as a single .ipynb file (please do not zip it when submitting to brightspace) including your answers to both the math and the experimental questions, in the correct order. Use markdown syntax to format your answers.

# Note: in solving the math questions, aim for general (symbolic) solutions and substitute the specific numbers at the end. This demonstrates a solid understanding of the key concepts. You can answer the math questions in two ways:  Use LaTeX to typeset the equations. Section H of [this LaTeX reference sheet](http://tug.ctan.org/info/latex-refsheet/LaTeXRefSheet.pdf) is a good reference. Here is another [LaTeX reference sheet](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference). The equations in the questions are typeset in LaTeX, so you can use them as examples.  Use neat handwriting, scan your solution using [AdobeScan](https://acrobat.adobe.com/ca/en/mobile/scanner-app.html), or [Dropbox](https://www.dropbox.com/doc-scanner-app) on your mobile phone, upload the image file, and embed it in your solution notebook. To this end (1) create an empty Markdown cell. 2) Drag-and-drop the image file into the empty Markdown cell, or click on the image icon at the top of the cell and select the image file. The Markdown code that will embed the image then appears.

# Your answers to the experimental questions should be in your solution notebook, in the form of code and text cells, using markdown for your text responses. You should also include the results of running your code. This means that you must not clear the output produced by your program.

# The marking criteria are described in rubrics. There are two rubrics, for math questions, and for experimental questions, respectively.

# You can submit multiple editions of your assignment. Only the last one will be marked. It is recommended to upload a complete submission, even if you are still improving it, so that you have something into the system if your computer fails for whatever reason.  IMPORTANT: PLEASE NAME YOUR PYTHON NOTEBOOK FILE AS:  <LASTNAME-<FIRSTNAME-Assignment-N.ipynb  for example: Axel-Soto-Assignment-1.ipynb \\

# ==============================================================================
# Section 2: 1. Vectors, hyperplanes and projections (8 pts)
# ==============================================================================

# Consider a vector space of two dimensions $(x1, x2)$, a point $A=(2,1)$ and a vector $\mathbf{v} = (3, 4)$.  a) What is the point defined by $\mathbf{v}$ considered as a position vector? If you move by one unit of length from point $A$ in the direction of $v$, what is the new point $B$ you will arrive at?

# $B = A + Norm(V)$  $\therefore B = <2,1 + \frac{<3,4}{\sqrt(3^2 + 4^2)} = <\frac{13}{5}, \frac{9}{5} $

# b) What is the position vector of a point $P$ derived by moving from point $A$ along $\mathbf{v}$ by a distance $s$? The result is a parametric representation of a line, where the parameter is $s$. The line contains $A$ and it is parallel to vector $\mathbf{v}$.

# $ P = A + s(Norm(v)) $  $ \therefore P = <1,2 + s(\frac{<3,4}{\sqrt(3^2 + 4^2)}) $

# c) Find vector $\mathbf{u}$ that is perpendicular to vector $\mathbf{v}$ above.

# $ u \cdot v = 0 $  $ <u0, u1 \cdot <v0,v1 = 0 $  $ u0 \cdot v0 + u1 \cdot v1 = 0 $  Let $u0 = c, c\in\mathbb{Z}$  $\therefore u1 = \frac{-(c \cdot v0)}{v1}$  $\therefore <c, u1$  Sample Solution:  Let ${u0} = 4$  $\therefore u1 = \frac{-(4 \cdot 3)}{4} = \frac{-12}{4} = -3$  $\therefore u = <4, -3$

# d) Given point $A$ and vector $\mathbf{u}$, provide a vector equation that every point $P$ on the line must satisfy. Reduce this equation to the form $ax1 + bx2 = 1$, i.e. calculate $a$ and $b$ in terms of $A$ and $\mathbf{u}$. $x1, x2$ are the coordinates of $P$. \\ Hint: the inner product of vector $\mathbf{u}$ and a vector parallel to the line is zero.

# Let $u = <u0, u1$  Let $P = <x0, x1$  $ (P - A) \cdot <u0, u1 = 0$  $ <x0 - a0, x1 - a1 \cdot <u0, u1 = 0 $  $ (x0 - a0) \cdot u0 + (x1 - a1) \cdot u1 = 0$  $ x0 \cdot u0 - a0 \cdot u0 + x1 \cdot u1 - a1 \cdot u1 = 0$  $ x0 \cdot u0 + x1 \cdot u1 = a0 \cdot u0 + a1 \cdot u1$  $ x0 \cdot \frac{u0}{a0 \cdot u0 + a1 \cdot u1} + x1 \cdot \frac{u1}{a0 \cdot u0 + a1 \cdot u1} = 1$  $ \therefore x0 \cdot \frac{4}{5} + x1 \cdot \frac{-3}{5} = 1$

# e) Generalize part (d) to a plane in three dimensions, i.e. given point $A$ and vector $\mathbf{u}$, provide a vector equation that every point $P$ on the plane must satisfy. Can you reduce this equation to the form $ax1 + bx2 + cx3 = 1$, where $x1, x2, x3$ are the coordinates of $P$? \\ Hint: the inner product of vector $\mathbf{u}$ with a vector parallel to the plane is zero.

# Let $u = <u0, u1, u2$  Let $P = <x0, x1, x2$  $ (P - A) \cdot <u0, u1,u2 = 0$  $ <x0 - a0, x1 - a1, x2 - a2 \cdot <u0, u1,u2 = 0 $  $ (x0 - a0) \cdot u0 + (x1 - a1) \cdot u1 + (x2 - a2) \cdot u2 = 0$  $ x0 \cdot u0 - a0 \cdot u0 + x1 \cdot u1 - a1 \cdot u1 + u2 \cdot x2 \cdot u2 - a2 \cdot u2 = 0$  $ x0 \cdot u0 + x1 \cdot u1 + x2 \cdot u2 = a0 \cdot u0 + a1 \cdot u1 + a2 \cdot u2$  $ x0 \cdot \frac{u0}{a0 \cdot u0 + a1 \cdot u1 + a2 \cdot u2$} + x1 \cdot \frac{u1}{a0 \cdot u0 + a1 \cdot u1 + a2 \cdot u2$} + x1 \cdot \frac{u2}{a0 \cdot u0 + a1 \cdot u1 + a2 \cdot u2$} = 1$  $ \therefore x0 \cdot \frac{4}{5} + x1 \cdot \frac{-3}{5} + x1 \cdot \frac{0}{5} = 1$

# f) Consider a plane in three dimensions defined by a point $A$ and a normal vector $\mathbf{u}$. Given a point $B$ not on the plane, find the projection $B'$ of point $B$ onto the plane. Hint: You can make use of two properties about projection $B'$. One property is based on vector $B' - A$ being parallel to the plane. Another property is based on vector $B' - B$ being parallel to vector $\mathbf{u}$.

# Let $v = <b0 - a0, b1 - a1, b2 - a2$  $v = v\perp + v\parallel$  $v\parallel = \frac{v \cdot u}{||u||^2} \cdot u$  $v\perp = v - v\parallel$  $\therefore B' = A + v\perp$

# ==============================================================================
# Section 3: 2.Vectorized gradient descent (12 pts)
# ==============================================================================

# %% Cell 0
import numpy as np
from sklearn import datasets
import plotly.express as px

# In this question we are going to learn first hand how gradient descent works in the context of a toy dataset. Note that we are not making use of a validation strategy here by using a test set or cross-fold validation—which is something that would be otherwise generally recommended. This exercise focuses on the inner workings of gradient descent applied to the quadratic cost function that was learned in the Linear regression class. Note : It’s important that the code provided is well documented and that no other external package is used (pandas, numpy are ok).

# %% Cell 1
X,y = datasets.make_regression(100,2,random_state= 42)
print(f'X shape: {X.shape}\ny shape:{y.shape}')

# a) Given the values of X, we want to fit a linear regression model to predict the y values. We will implement a vectorized version of the gradient descent algorithm. Input (X) and target (y) are provided as arguments. X is modified (Xb) to account for the bias. The coefficients (theta) have been initialized for you. The shape of the variables X, y, b have been given, use this as a guide to output appropriate shape.

# Fill in the missing variables inside the gradient descent iteration loop and return the updated costlist and parameters.

# %% Cell 2
def LR(X, y, lr, iterations):
    # Initializations

    # Adding 1 column in X for bias
    b = np.ones((len(X), 1))             # shape : (N,1)
    X_b = np.append(X, b, axis=1)        # Shape : (N, features+1)
    theta = np.zeros((X_b.shape[1], 1))  # shape : (features+1 ,1)
    y = y.reshape(-1, 1)                 # shape : (N,1)
    cost_list = []

    # Gradient Descent
    # Fill Code Below
    for _ in range(iterations):
        # Compute y using initial values of parameters
        y_pred = X_b.dot(theta)

        residual = y_pred - y
        # Compute loss
        loss = (residual**2) / 2

        # Compute the cost and append to cost_list. This would be later used for plotting.
        cost = np.sum(loss)
        cost_list.append(cost)

        # Compute the average gradient of loss function with respect to theta.
        grad = X_b.T.dot(residual) / len(X_b)

        # Perform an optimization step on the parameters
        theta -= lr * grad

    return cost_list, theta

# %% Cell 3
# You can modify learning_rate and number of iterations as required.
cost_list,theta = LR(X,y,0.002,4500) # 

# %% Cell 4
def plot_losses(cost_list):
  """This function plots the cost_list"""
  fig = px.line(y= cost_list)
  fig.update_layout(title= "Loss vs Iteration",xaxis_title='Iteration',yaxis_title='Cost')
  return fig

# %% Cell 5
# Run this cell to view how loss changes with iterations
plot_losses(cost_list)

# Below is a function that takes as input the parameters($\theta$) and Input (X). Fill in the missing code to output the y value using the input parameters.

# %% Cell 6
# theta shape : (features+1 ,1)
def predict(theta, x):
  return x.dot(theta)

# b) Compare this model with a solution computed in closed form. Input (X) and target (y) are provided as arguments. X is modified (Xb) to account for the bias. The function should return the optimum parameters($\theta$).

# %% Cell 7
from numpy.linalg import inv
def linear_direct(X,y):

  #initializations
  b = np.ones((len(X),1)) # shape : (N,1)
  X_b = np.append(X,b,axis=1) # Shape : (N, features+1)
  
  # Fill code below
  theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

  return theta

# c) Discuss how the choice of learningrate and number of iterations in part (a) affects the fitting of the model?

# %% Cell 8
# Low learning, Low iterations.
cost_list_0, theta_0 = LR(X, y, 0.001, 100)
print("low low")
plot_losses(cost_list_0).show()
# Low learning, High iterations.
cost_list_1, theta_1 = LR(X, y, 0.001, 10000)
print("low high")
plot_losses(cost_list_1).show()
# High learning, Low iterations.
cost_list_2, theta_2 = LR(X, y, 0.1, 100)
print("high low")
plot_losses(cost_list_2).show()
# High learning, High iterations
cost_list_3, theta_3 = LR(X, y, 0.1, 10000)
print("high high")
plot_losses(cost_list_3).show()

# Answer Here:  | | Low no. Iterations | High no. Iterations | |------------------|------------------------|-------------------------| |Low learning Rate | Underfits | Overfits Slow | |High Learning Rate| Overfits Quick | Overfits Extremely Quick|

# ==============================================================================
# Section 4: 3.Classification (10 pts)
# ==============================================================================

# In this question, you will experiment with different classification algorithms from Sklearn API on the [Wine dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/).

# %% Cell 9
X,y = datasets.load_wine(return_X_y=True)

# %% Cell 10
#Uncomment to know more about the dataset
#print(datasets.load_wine().DESCR)

# %% Cell 11
import matplotlib.pyplot as plt
import pandas as pd
 # This shows the counts of labels 
def plot_distribution(y):
    plt.hist(y)
    plt.xticks([0,1,2],['Class_0','Class_1','Class_2'])
    plt.title('Distribution of labels')
    plt.ylabel("frequency")
    plt.show()

plot_distribution(y)

# %% Cell 12
pd.DataFrame(y).value_counts()

# a) First your task is to divide the above dataset into 3 splits (i.e. training, validation and test sets) such that each split follows approximately the same distribution of labels as the original dataset.The distribution of labels in the original dataset has been plotted for you above. The aim of this question is for you to read the [scikit-learn API](https://scikit-learn.org/stable/userguide.html) and get comfortable with training/validation splits.

# Implement the following function to return 3 data splits in the ratio 7:1:2 (namely the training set, validation set and test set respectively) such that each split follows a similar distribution of labels.  Hint : [traintestsplit](https://scikit-learn.org/stable/modules/generated/sklearn.modelselection.traintestsplit.html) is a helpful sklearn function for this task.

# %% Cell 13
from sklearn import model_selection

def return_splits(X,y)-> (tuple):
  """This function should return three tuples, one for each split.
     where each tuple should contain (X_split,y_split) respectively. """

  # Code below
  train_set_x, test_set_x, train_set_y, test_set_y = model_selection.train_test_split(X, y, test_size=0.2, random_state=1, stratify=y);
  train_set_x, validation_set_x, train_set_y, validation_set_y = model_selection.train_test_split(train_set_x, train_set_y, test_size=(1/8), random_state=1, stratify=train_set_y)
  train_set,validation_set,test_set = [(train_set_x, train_set_y), (validation_set_x, validation_set_y), (test_set_x, test_set_y)]

  return train_set,validation_set,test_set

# %% Cell 14
#Run this cell to get your splits and plot the distribution of labels for each split.

train_set,validation_set,test_set = return_splits(X,y)

# The 3 plots should look similar.
plot_distribution(train_set[1])
plot_distribution(test_set[1])
plot_distribution(validation_set[1])

# In this question you will experiment with two traditional classification models.The objective of this question is to demonstrate your ability to compare different machine learning models, and derive a conclusion if possible.  The models are:  [SVM Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).  Here we will use a different method to validate models, which is called k-fold cross validation ([Stratified K-fold](https://scikit-learn.org/stable/modules/generated/sklearn.modelselection.StratifiedKFold.html#sklearn.modelselection.StratifiedKFold)).

# Note: Since, here we would use k-fold cross validation. You do not need to use the splits defined in the previous question.

# %% Cell 15
X,y = datasets.load_wine(return_X_y=True)

# b) For the SVM classifier, use the default parameters, and 5-fold cross validation, and report the overall accuracy and confusion matrix, as well as accuracy and confusion matrix for each fold. What is the standard deviation of accuracy over the folds?

# %% Cell 16
from sklearn import pipeline, preprocessing, svm, metrics

svm_clf = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.SVC(gamma="auto"))

def get_k_cv(X, y, clf, k = 5, print_individual = False):
  skf = model_selection.StratifiedKFold(n_splits=k)

  no_of_labels = len(np.unique(y))
  accuracy_scores = np.empty(shape=(k, ))
  confusion_matrix_total = np.zeros(shape=(no_of_labels, no_of_labels))
  
  fold = 0;
  for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix_total += confusion_matrix
    accuracy_scores[fold] = accuracy_score

    if print_individual:
        print(f'Fold {fold+1}:')
        print(f'Confusion Matrix\n', confusion_matrix)
        print(f'Accuracy Score\n', accuracy_score)
        print("------------------------------")
    
    fold += 1

  return (confusion_matrix_total, accuracy_scores)

def print_total(confusion_matrix_total, accuracy_scores, folds = 5) -> None:
  print(f'Confusion Matrix Mean: \n{confusion_matrix_total / folds} \n')
  print(f'Accuracy Mean: \n{accuracy_scores.mean():.2f}')
  print(f'Accuracy Standard Deviation: \n{accuracy_scores.std():.2f}')

print_total(*get_k_cv(X, y, svm_clf, print_individual=True))

# c) For the Decision tree classifier, experiment with different numbers of max depths of the tree (range from 2 to 8).Experiment with other parameters as needed. Evaluate each parameter selection using 5-fold cross validation, and report the overall accuracy and the confusion matrix. Only report the interesting parameter settings. Hint: [Grid Search CV](https://scikit-learn.org/stable/modules/generated/sklearn.modelselection.GridSearchCV.html) is a helpful utility for this task.

# %% Cell 17
from sklearn import tree

"""
@param max_depth: A low max_depth can result in underfitting whereas a high max_depth can result in overfitting.
@param max_features: A low amount of features will likely result in underfitted models 
               whereas a large amount of features will result in excessive computation power usage
@param min_impurity_decrease: Determines the split threshold
"""
clf_parameters = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
    "max_features": [1, 2, 3, 4, 5, 6, 7, 8, "sqrt", "log2"],
    "min_impurity_decrease": [0.1, 0.2, 5, 2, 3, 100],
}

clf_search = model_selection.GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=clf_parameters)
clf_search.fit(X, y)

dt_clf = tree.DecisionTreeClassifier(**clf_search.best_params_)

print_total(*get_k_cv(X, y, dt_clf))

# d) Summarize your findings from (a) and (b). What is the best performing classifier on the iris data set, considering the mean and standard deviation of accuracy of the two classifiers?

# Answer here:    The SVM classifier performs significantly better than the decision tree classifier.  Not only was the SVM classifier better performing in terms of accuracy, it was also less likely to misclasify samples.

# ==============================================================================
# Section 5: 4. Polynomial regression (10 pts)
# ==============================================================================

# This question aims at applying polynomial regression (generating [Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) followed by [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linearmodel.LinearRegression.html)) on a large data set and deciding the set of hyperparameters that best applies to this scenario.  a) Use the [Boston house-prices dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.loadboston.html#sklearn.datasets.loadboston) from scikit-learn and apply a polynomial regression with the full set of features. Experiment with polynomials of different degrees. Compare their performance with each other using a fixed training and testing partition (where the training set is 80% of the data set and the remainder for testing).

# %% Cell 18
from sklearn import linear_model

def plot_learning_curve(clf, X, y, degree):
  X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, random_state=2)
  no_iterations = len(X_train)
  train_error, test_error = np.empty(shape=(no_iterations,)), np.empty(shape=(no_iterations,))

  # Gather training and testing error based on different set cardinalities.
  for train_size in range(1, no_iterations):
    X_train_act, y_train_act = X_train[:train_size], y_train[:train_size]
    clf.fit(X_train_act, y_train_act)
    y_train_pred, y_test_pred = clf.predict(X_train_act), clf.predict(X_test)
    train_error[train_size-1] = metrics.mean_squared_error(y_train_act, y_train_pred)
    test_error[train_size-1] = metrics.mean_squared_error(y_test, y_test_pred)
  
  # Plotting.
  fig, ax = plt.subplots()
  fig.set_size_inches(10, 12)
  ax.plot(np.sqrt(train_error), "r-+", linewidth=2, label="Training error")
  ax.plot(np.sqrt(test_error), "b-", linewidth=3, label="Testing Error")
  ax.set_xlabel("Training set size")
  ax.set_ylabel("RSME")
  ax.set_title(f'Polynomial Regree Degree {degree}')
  ax.legend()
  ax.set_ylim([0, 50])
 
  return fig

X,y = datasets.load_boston(return_X_y=True)
preprocessing.Normalizer().transform(X)

for degree in range(1, 6):
  clf = pipeline.Pipeline([
                         ("poly_features", preprocessing.PolynomialFeatures(degree, include_bias=True)), 
                         ("lin_reg", linear_model.LinearRegression())
                      ])
  
  plot_learning_curve(clf, X, y, degree).show()

# b)Discuss the interpretation of the results. Use visualizations of appropriate quantities to make sense of the results and support your interpretation.

# %% Cell 19
# Refer to part 4A (code is modularized there).

# Discussion Here:

# Through the analysis of the learning curves for both the training and test sets on different degree polynomial regression models, there are a few general conclusions we can make in relation to our current data set.  1. On higher degree polynomial regression models (degree  2), we will typically see less erroneous behaviour during training but more erroneous behaviour during testing as a result of overfitting. 2. On lower degree polynomial regression models (degree < 2), we will typically see similar erroneous behaviour between training and testing sets as a result of underfitting. 3. The optimal degree for the polynomial regression model is 2.
