# Implement a function that finds the k most similar documents to a query document
# using cosine similarity.

# for future information, there's a few ways to optimize performance
# I think it's also possible to normalize vectors ahead of time and store them
# which allows cosine similarity to be computed via dot product only
# first, using a matrix / linear algebra to perform the batch operations
# then, you can use ANN algorithms like HNSW (via Faiss)
from math import sqrt
from functools import lru_cache


def derive_norm(vector: list[float]) -> float:
    return sqrt(sum(x**2 for x in vector))


@lru_cache(maxsize=None)
def normalize_vec(vector: list[float]):
    if norm := derive_norm(vector) != 0.0:
        return [x / norm for x in vector]

    return vector


def pairwise_dot_product(vec1: list[float], vec2: list[float]) -> float:
    return sum(x * y for x, y in zip(vec1, vec2))


# a * b / [norm (a) * norm (b) == 1]
def cosine_similarity(vec1: list[float], vec2: list[float]):
    return pairwise_dot_product(normalize_vec(vec1), normalize_vec(vec2))


# Signature:
def find_similar_documents(
    query_vector: list[float], document_vectors: list[list[float]], k: int
) -> list[int]:
    """
    Returns indices of k most similar documents to query_vector.

    Args:
        query_vector: TF-IDF vector of the query document
        document_vectors: list of TF-IDF vectors for all documents
        k: Number of similar documents to return

    Returns:
        list of indices of the k most similar documents,
        ordered by similarity (most similar first)
    """
    # pre-compute normalized vectors if needed (optional optimization)
    for vec in document_vectors:
        normalize_vec(vec)

    # note: optimize via parallel processing if needed
    similarities = [
        cosine_similarity(query_vector, doc_vector) for doc_vector in document_vectors
    ]

    # heap is an interesting structure as the max can be calculated in O(1) and insertion is O(log n)
    # the maxs are also very simple to retrieve

    heap

    for idx, doc_vector in enumerate(document_vectors):
        similarity = cosine_similarity(query_vector, doc_vector)

    # note: we could create a heap and insert items to avoid full sort


# Example:
# query = [0.5, 0.3, 0.2]
# docs = [
#     [0.6, 0.2, 0.2],  # doc 0
#     [0.1, 0.8, 0.1],  # doc 1
#     [0.5, 0.3, 0.2],  # doc 2 (identical to query)
# ]
# find_similar_documents(query, docs, k=2)
# -> [2, 0]  (doc 2 most similar, then doc 0)
