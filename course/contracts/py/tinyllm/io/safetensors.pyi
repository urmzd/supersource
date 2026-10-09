# contracts/py/tinyllm/io/safetensors.pyi (L0.0 v0: F32 only; L0.6 takes the unit over and adds every dtype)
# chapter: ml/08-tinyllm/p00-foundations/00-byte-bigram.md
#
# The file format and the canonical writer rules are formats/safetensors.md.
from numpy.typing import NDArray

def save_safetensors(
    path: str, tensors: dict[str, NDArray], meta: dict[str, str]
) -> None:
    """Write tensors and metadata to path, byte-identical to the pinned
    `safetensors` library (formats/safetensors.md, "Rules a canonical writer
    follows"). An empty meta omits __metadata__. Tensor bytes are row-major
    little-endian whatever the array's memory order or byte order.
    ValueError for a tensor whose dtype is not a 4-byte float (F32), a
    tensor named "__metadata__", or a meta key or value that is not a str."""

def load_safetensors(path: str) -> tuple[dict[str, NDArray], dict[str, str]]:
    """Read path into (tensors, meta), checking the five reader rules of
    formats/safetensors.md. Arrays are native float32, C-contiguous, writable.
    ValueError for any rule violation, and for any dtype other than F32
    (unsupported until L0.6)."""
