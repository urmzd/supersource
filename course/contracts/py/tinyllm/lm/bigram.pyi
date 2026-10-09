# contracts/py/tinyllm/lm/bigram.pyi (L0.0 v0; L0.5 takes the unit over and keeps this API)
# chapter: ml/08-tinyllm/p00-foundations/00-byte-bigram.md
#
# A bigram language model is a [V, V] table `weight`: row i holds the
# next-token logits after token i, and softmax(weight[i]) is the model's
# distribution over the next token. V is the vocabulary size (256 for the
# byte tokenizer, formats/tokenizer.md). Ids are integers in [0, V).
from numpy.typing import ArrayLike, NDArray

class BigramLM:
    weight: NDArray  # float32 [V, V]; saved as `bigram.weight` (formats/safetensors.md)

    def __init__(self, weight: ArrayLike | None = None) -> None:
        """Empty model, or one built from a [V, V] float32 table (a loaded checkpoint).
        ValueError unless weight is 2-D, square, float32, and finite."""

    @property
    def vocab_size(self) -> int:
        """V. RuntimeError before fit_counts or a weight."""

    def fit_counts(self, ids: ArrayLike, vocab_size: int, alpha: float = 1.0) -> None:
        """Count every adjacent pair (ids[t], ids[t + 1]) into C[V, V] and store
        weight[i, j] = log((C[i, j] + alpha) / (sum_k C[i, k] + alpha * V)) as float32.
        ValueError when ids is not 1-D integer, an id is outside [0, vocab_size),
        or alpha <= 0."""

    def logits(self, ids: ArrayLike) -> NDArray:
        """float32 [T, V] = onehot(ids) @ weight, computed by tl_matmul_f32 (M03.1)
        through the rt.01 ctypes loader; row t equals weight[ids[t]] exactly.
        ValueError for an id outside [0, V). The loader's errors pass through:
        TlError (a RuntimeError) when the C call fails, and OSError or
        AbiMismatch when libtinyllm cannot be loaded. There is no numpy fallback."""

    def nll(self, ids: ArrayLike) -> float:
        """Mean negative log-likelihood in nats per predicted token:
        -(1 / (T - 1)) * sum_t log softmax(logits(ids[:-1]))[t, ids[t + 1]].
        ValueError when len(ids) < 2."""

    def sample(
        self, prefix: list[int], n: int, temperature: float, seed: int
    ) -> list[int]:
        """Continue prefix with exactly n new ids and return only the new ids.
        temperature == 0 is greedy (highest logit, ties to the lowest id).
        Otherwise p = softmax(logits / temperature) in float64, one uniform u in
        [0, 1) per token from numpy.random.default_rng(seed), and the next id is
        the smallest j with cumsum(p)[j] > u (inverse CDF). Same seed, same ids.
        ValueError for an empty prefix, n < 0, or temperature < 0."""
