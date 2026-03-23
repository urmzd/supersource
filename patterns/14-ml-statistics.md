# ML & Statistics Patterns

## Core Insight

Most ML problems reduce to **"find parameters that minimize a loss function."** The differences lie in the model family (linear, probabilistic, neural), the loss function (MSE, cross-entropy, likelihood), and the optimization strategy (closed-form, gradient descent, EM).

## Pattern 1: Linear Regression (Closed-Form & Gradient Descent)

**When to use**: Predicting a continuous target from features with an approximately linear relationship.

**Closed-form (Normal Equation)**:
```
w = (X^T X)^{-1} X^T y
```

**Gradient descent update**:
```python
for each epoch:
    predictions = X @ w + b
    error = predictions - y
    w -= lr * (1/n) * X.T @ error
    b -= lr * (1/n) * sum(error)
```

**Key decisions**: Feature normalization, learning rate, regularization (Ridge L2 vs Lasso L1).

**Interview problems**: Implement linear regression from scratch, derive the normal equation, compare Ridge vs Lasso.

## Pattern 2: Logistic Regression & Binary Classification

**When to use**: Binary classification with a probabilistic interpretation.

**Template**:
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for each epoch:
    z = X @ w + b
    a = sigmoid(z)
    dw = (1/n) * X.T @ (a - y)       # cross-entropy gradient
    db = (1/n) * np.sum(a - y)
    w -= lr * dw
    b -= lr * db
```

**Loss**: Binary cross-entropy: `L = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]`

**Interview problems**: Derive gradient of cross-entropy, implement logistic regression, explain decision boundary geometry.

## Pattern 3: K-Means Clustering

**When to use**: Unsupervised grouping into k clusters when clusters are roughly spherical.

**Template**:
```python
centroids = random_init(k)
while not converged:
    # Assign each point to nearest centroid
    assignments = [argmin(dist(x, c) for c in centroids) for x in data]
    # Recompute centroids
    centroids = [mean(data[assignments == i]) for i in range(k)]
```

**Convergence criterion**: Centroids stop moving (or move below a threshold).

**Key decisions**: Number of clusters (elbow method, silhouette score), initialization (k-means++).

**Interview problems**: Implement K-means, explain failure modes (elongated clusters, varying sizes), compare with GMM.

## Pattern 4: Naive Bayes & Generative Models

**When to use**: Classification via Bayes' theorem, especially with categorical features or text.

**Template**:
```
P(class | features) ∝ P(class) * Π P(feature_i | class)

# For Gaussian Naive Bayes:
P(x_i | class) = N(x_i; μ_class, σ²_class)
```

**Why "naive"**: Assumes feature independence given the class — often wrong but surprisingly effective.

**Interview problems**: Spam classification, explain generative vs discriminative models, derive MAP estimate.

## Pattern 5: Gaussian Mixture Models (EM Algorithm)

**When to use**: Soft clustering where each point has a probability of belonging to each cluster.

**Template (EM)**:
```
E-step: Compute responsibilities
    r(i,k) = π_k * N(x_i | μ_k, Σ_k) / Σ_j π_j * N(x_i | μ_j, Σ_j)

M-step: Update parameters
    μ_k = Σ r(i,k) * x_i / Σ r(i,k)
    Σ_k = weighted covariance
    π_k = Σ r(i,k) / N
```

**vs K-means**: GMM generalizes K-means (K-means is EM with hard assignments and spherical covariance).

**Interview problems**: Derive EM updates, explain convergence guarantees, compare with K-means.

## Pattern 6: Neural Networks & Backpropagation

**When to use**: Complex nonlinear relationships, high-dimensional inputs (images, text, sequences).

**Template (forward + backward)**:
```python
# Forward pass
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
output = softmax(z2)

# Backward pass (chain rule)
dz2 = output - y_onehot
dW2 = a1.T @ dz2
dz1 = (dz2 @ W2.T) * relu_derivative(z1)
dW1 = X.T @ dz1
```

**Key concepts**: Vanishing/exploding gradients, batch normalization, dropout, Adam optimizer.

**Interview problems**: Implement a 2-layer network from scratch, derive backprop for softmax + cross-entropy, explain vanishing gradients.

## Pattern 7: Feature Selection & Evaluation

**When to use**: Every supervised learning problem — choosing features and measuring model quality.

**Feature selection**:
```python
# Pearson correlation for feature importance
correlations = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]
top_features = argsort(abs(correlations))[-k:]
```

**Evaluation metrics**:
- Regression: MSE, RMSE, R²
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Clustering: Silhouette score, inertia (within-cluster sum of squares)

**Interview problems**: Explain bias-variance tradeoff, when to use which metric, cross-validation strategies.

## Company Targeting

| Company | Focus | Difficulty |
|---------|-------|------------|
| Google | Neural nets, optimization theory, large-scale ML | Hard |
| Meta | Recommendation systems, logistic regression at scale | Medium-Hard |
| Amazon | Applied ML (classification, regression), A/B testing | Medium |
| Netflix | Clustering, collaborative filtering, evaluation metrics | Medium-Hard |
| Stripe | Anomaly detection, logistic regression for fraud | Medium |

## Algorithm Comparison

| Algorithm | Type | Loss/Objective | Strengths | Weaknesses |
|-----------|------|---------------|-----------|------------|
| Linear Regression | Supervised | MSE | Interpretable, fast | Linear only |
| Logistic Regression | Supervised | Cross-entropy | Probabilistic, interpretable | Linear boundary |
| K-Means | Unsupervised | Inertia | Simple, scalable | Assumes spherical clusters |
| Naive Bayes | Supervised | Log-likelihood | Fast, works with small data | Independence assumption |
| GMM | Unsupervised | Log-likelihood | Soft assignments, flexible | Sensitive to initialization |
| Neural Network | Supervised | Task-dependent | Universal approximator | Data-hungry, black box |
