# Extracted from clustering.ipynb
# NOTE: Code cells preserved; markdown condensed into section headers/comments

# ==============================================================================
# Section 1: Notebook
# ==============================================================================

# Colab source: CSCI4155 Assignment 2 (see previous commit machine-learning/clustering.ipynb)

# ==============================================================================
# Section 2: Assignment 2
# ==============================================================================

# Assignment 2: clustering exercises and analysis.

# ==============================================================================
# Section 3: Part 1: Synthetic dataset
# ==============================================================================

# Question 1 Start by generating a 2D dataset that has 3 Gaussian clusters. The first two should be as in the cluster creation exercise from class, i.e., cluster 1 should be centred at (10,10) and be spherical and cluster 2 should be centred at (0,0) and have contours in the shape of an ellipse (with a radius ratio of roughly 4:1), while the third cluster can be whatever you'd like as long as it is distinct from the first two but still has some overlap with one of them. Include a 1000 datapoints for each cluster.

# %% Cell 0
import numpy as np
import matplotlib.pyplot as plt

cluster_1 = np.random.normal((10, 10), (1, 1), (1000, 1, 2))
cluster_2 = np.random.normal((0, 0), (4, 1), (1000, 1, 2))
cluster_3 = np.random.normal((5, 5), (1.25, 2), (1000, 1, 2))
all_clusters = np.vstack([cluster_1, cluster_2, cluster_3])
print(all_clusters.shape)

# Question 2 Produce a scatter plot of your clusters, assigning each cluster a different color.

# %% Cell 1
fig, ax = plt.subplots()
ax.scatter(cluster_1[..., 0], cluster_1[..., 1], label="cluster 1", c="red")
ax.scatter(cluster_2[..., 0], cluster_2[..., 1], label="cluster 2", c="blue")
ax.scatter(cluster_3[..., 0], cluster_3[..., 1], label="cluster 3", c="green")
ax.legend()

# ==============================================================================
# Section 4: Part 2: Clustering
# ==============================================================================

# Suppose you didn't know to which cluster each datapoint belonged to, and wanted to find out. One of the first things you might try is the following algorithm, where $k$ is a hyperparameters controlling the number of clusters we're trying to find:  1. Initialize the k cluster centroids, $\mu1, \mu2, \ldots, \muk$, randomly. 2. Repeat until convergence (Question 1: what should the convergence criterion be?): 1. Assign each datapoint to the cluster with the nearest centroid; 2. Re-calculate the cluster centroids, i.e. the mean of all datapoints assigned to each cluster

# ==============================================================================
# Section 5: Question 1 Answer
# ==============================================================================

# Question 2: Implement this in Python. Define a Python class $\it{clustering}$, that takes as initialization parameters $\it{n{clusters}}$, the number of clusters, $\it{n{iter}}$, the maximum number of iterations of the above algorithm, and any other parameters you feel are needed to check for convergence (if any). The cluster centroids should be a class attribute. Define two methods, $\it{fit}$, taking in a set of points, $\mathbf{X}$ and performing clustering using the algorithm above, and $\it{predict}$, that takes in a set of points and returns a prediction for which cluster those points belong to.

# %% Cell 2
from dataclasses import dataclass, field


def get_distance(p, q):
    """Returns the Euclidean distance between `p` and `q`."""
    return np.sum(np.power((p - q), 2), axis=-1)


def get_k_centroids(k=3, lb=-20.0, ub=20.0) -> np.ndarray:
    """Returns K centroids somewhere in the range (lb,ub] in the shape (k, 2)"""
    return np.random.uniform(lb, ub, (k, 2))


# print(get_distance(get_k_centroids(), all_clusters))


@dataclass
class NaiveKClustering:
    n_clusters: int = 3
    n_iters: int = 50
    centroids: np.ndarray = field(default_factory=get_k_centroids)
    min_distance_update_threshold: float = 1e-20

    def fit(self, X: np.ndarray) -> None:
        prev_centroids = self.centroids
        for _ in range(self.n_iters):
            pred_clusters, distances = self.predict(
                X, centroids=prev_centroids, return_distances=True
            )
            clusters = [[] for _ in range(prev_centroids.shape[0])]

            # Group all points by their predicted cluster.
            for point_idx, cluster_idx in enumerate(pred_clusters):
                clusters[cluster_idx].append(X[point_idx])

            # Stop if the centroids have failed to update significantly.
            new_centroids = np.array(
                [
                    np.average(np.vstack(cluster), axis=0)
                    if len(cluster) > 0
                    else np.zeros(
                        2,
                    )
                    for cluster in (clusters)
                ]
            )

            # print(new_centroids)
            prev_centroids = new_centroids

        self.centroids = prev_centroids
        # print("ALL ITERATIONS USED!")

    def predict(self, X, centroids=None, return_distances=False) -> np.ndarray:
        centroids_to_use = self.centroids if centroids is None else centroids
        # Calculate the distance between the points and centroids.
        distances = get_distance(centroids_to_use, X)

        # A point is associated to the closest centroid.
        pred_clusters = np.argmin(distances, axis=-1)

        if return_distances:
            return pred_clusters, distances

        return pred_clusters


# Question 3: Use your clustering class on the synthetic dataset you created in Part 1. Comment on how well it clusters your dataset.


# %% Cell 3
@dataclass
class RunData:
    final_centroids: np.ndarray
    inital_centroids: np.ndarray
    cluster_1: np.ndarray
    cluster_2: np.ndarray
    cluster_3: np.ndarray


def map_array_to_colours(X, colours=["red", "green", "blue"]):
    return np.array(list(map(lambda index: colours[index], X)))


def run_instance() -> RunData:
    model = NaiveKClustering()
    intial_centroids = model.centroids
    model.fit(all_clusters)
    cluster_1_preds = model.predict(cluster_1)
    cluster_2_preds = model.predict(cluster_2)
    cluster_3_preds = model.predict(cluster_3)

    run_data = RunData(
        final_centroids=model.centroids,
        inital_centroids=intial_centroids,
        cluster_1=cluster_1_preds,
        cluster_2=cluster_2_preds,
        cluster_3=cluster_3_preds,
    )

    return run_data


def plot_run_instance(instance: RunData):
    _, ax = plt.subplots()
    print(f"INITIAL_CENTROIDS: {instance.inital_centroids}")
    ax.scatter(
        cluster_1[..., 0], cluster_1[..., 1], c=map_array_to_colours(instance.cluster_1)
    )
    ax.scatter(
        cluster_2[..., 0], cluster_2[..., 1], c=map_array_to_colours(instance.cluster_2)
    )
    ax.scatter(
        cluster_3[..., 0], cluster_3[..., 1], c=map_array_to_colours(instance.cluster_3)
    )
    print(f"FINAL_CENTROIDS: {instance.final_centroids}")
    print("\n")


plot_run_instance(run_instance())

# ==============================================================================
# Section 6: Conclusion
# ==============================================================================

# Question 4: How do the found clusters vary as you run the algorithm again?

# %% Cell 4
for _ in range(5):
    instance = run_instance()
    plot_run_instance(instance)

# ==============================================================================
# Section 7: Conclusion
# ==============================================================================

# The model clusters the points relatively well. Confusion appears mainly where
# clusters overlap, but the predicted cluster shapes resemble the true clusters.
#
# Rerunning the algorithm does not significantly change the cluster shapes.
