"""Basic layers with PyTorch's names and shapes (L0.4).

Every forward is a composition of op-library calls (L0.2), so backward needs
no code here. Initial weights come from M07.3's initializers, drawn from a
PCG32 (M06.3) in parameter registration order.

Contract: contracts/py/tinyllm/nn/layers.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.init import normal_init
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32


def _rng(rng: Any, purpose: str) -> Any:
    """rng, or the default stream for `purpose` (spec/pcg32.md sub-streams)."""
    # SOLUTION-BEGIN L0.4
    return rng if rng is not None else PCG32(0).substream(purpose)
    # SOLUTION-END


def _param(a: ArrayLike) -> Tensor:
    # SOLUTION-BEGIN L0.4
    return Tensor(np.asarray(a, dtype=np.float32), requires_grad=True)
    # SOLUTION-END


class Linear(Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True, rng: Any = None) -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        if in_f < 1 or out_f < 1:
            raise ValueError(f"Linear({in_f}, {out_f}): sizes must be positive")
        r = _rng(rng, "init")
        # torch's layout: one row per output, so y = x @ W^T.
        self.weight = _param(normal_init((out_f, in_f), 1.0 / math.sqrt(in_f), r))
        self.bias = _param(np.zeros(out_f)) if bias else None
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L0.4
        y = F.matmul(x, F.transpose(self.weight, 0, 1))
        return y + self.bias if self.bias is not None else y
        # SOLUTION-END


class Embedding(Module):
    def __init__(self, n: int, d: int, rng: Any = None) -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        self.weight = _param(normal_init((n, d), 1.0, _rng(rng, "init")))
        # SOLUTION-END

    def forward(self, ids: ArrayLike) -> Tensor:
        # SOLUTION-BEGIN L0.4
        return F.embedding(self.weight, ids)
        # SOLUTION-END


class LayerNorm(Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        self.eps = float(eps)
        self.weight = _param(np.ones(d))
        self.bias = _param(np.zeros(d))
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L0.4
        mu = F.mean(x, axis=-1, keepdims=True)
        var = F.var(x, axis=-1, keepdims=True)  # population variance, as torch
        xhat = (x - mu) * (var + self.eps) ** -0.5
        return xhat * self.weight + self.bias
        # SOLUTION-END


class Dropout(Module):
    def __init__(self, p: float, rng: Any = None) -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout p must be in [0, 1], got {p}")
        self.p = float(p)
        self.rng = _rng(rng, "dropout")
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L0.4
        return F.dropout(x, self.p, self.training, self.rng)
        # SOLUTION-END


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L0.4
        return F.relu(x)
        # SOLUTION-END


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L0.4
        return F.tanh(x)
        # SOLUTION-END


class GELU(Module):
    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
        self.approximate = approximate
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L0.4
        return F.gelu(x, approximate=self.approximate)
        # SOLUTION-END


class Sequential(Module):
    def __init__(self, *mods: Module) -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        for i, m in enumerate(mods):
            if not isinstance(m, Module):
                raise TypeError(f"Sequential item {i} is {type(m).__name__}, not a Module")
            setattr(self, str(i), m)  # names "0", "1", ...: torch's state_dict keys
        # SOLUTION-END

    def forward(self, x: Any) -> Any:
        # SOLUTION-BEGIN L0.4
        for m in self._children.values():
            x = m(x)
        return x
        # SOLUTION-END

    def __getitem__(self, i: int) -> Module:
        # SOLUTION-BEGIN L0.4
        return list(self._children.values())[i]
        # SOLUTION-END

    def __len__(self) -> int:
        # SOLUTION-BEGIN L0.4
        return len(self._children)
        # SOLUTION-END

    def __iter__(self) -> Iterator[Module]:
        # SOLUTION-BEGIN L0.4
        return iter(list(self._children.values()))
        # SOLUTION-END


class ModuleList(Module):
    def __init__(self, mods: Sequence[Module] = ()) -> None:
        # SOLUTION-BEGIN L0.4
        super().__init__()
        for m in mods:
            self.append(m)
        # SOLUTION-END

    def append(self, m: Module) -> "ModuleList":
        # SOLUTION-BEGIN L0.4
        if not isinstance(m, Module):
            raise TypeError(f"ModuleList holds Modules, got {type(m).__name__}")
        setattr(self, str(len(self._children)), m)
        return self
        # SOLUTION-END

    def __getitem__(self, i: int) -> Module:
        # SOLUTION-BEGIN L0.4
        return list(self._children.values())[i]
        # SOLUTION-END

    def __len__(self) -> int:
        # SOLUTION-BEGIN L0.4
        return len(self._children)
        # SOLUTION-END

    def __iter__(self) -> Iterator[Module]:
        # SOLUTION-BEGIN L0.4
        return iter(list(self._children.values()))
        # SOLUTION-END
