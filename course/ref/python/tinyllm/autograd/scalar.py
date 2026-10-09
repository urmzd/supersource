"""Scalar reverse-mode autodiff: the Value graph (M08.2).

Every operation records its inputs and a closure that applies its local
derivative. backward() seeds the output's gradient with 1 and walks the
graph once, output first, so each node's gradient is complete (every
consumer has added its share) before it is pushed further back.

Contract: contracts/py/tinyllm/autograd/scalar.pyi.
"""

from __future__ import annotations

import math

from tinyllm.autograd.graph import toposort


def _noop() -> None:
    return None


class Value:
    """A scalar and the gradient of the final output with respect to it."""

    def __init__(self, data: float, _children: tuple = (), _op: str = "") -> None:
        # SOLUTION-BEGIN M08.2
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = _noop
        # SOLUTION-END

    def __repr__(self) -> str:
        return f"Value(data={self.data!r}, grad={self.grad!r})"

    def __add__(self, other: Value | float) -> Value:
        # SOLUTION-BEGIN M08.2
        o = _lift(other)
        out = Value(self.data + o.data, (self, o), "+")

        def _backward() -> None:
            self.grad += out.grad
            o.grad += out.grad

        out._backward = _backward
        return out
        # SOLUTION-END

    def __radd__(self, other: float) -> Value:
        # SOLUTION-BEGIN M08.2
        return self + other
        # SOLUTION-END

    def __sub__(self, other: Value | float) -> Value:
        # SOLUTION-BEGIN M08.2
        return self + (-_lift(other))
        # SOLUTION-END

    def __rsub__(self, other: float) -> Value:
        # SOLUTION-BEGIN M08.2
        return _lift(other) + (-self)
        # SOLUTION-END

    def __mul__(self, other: Value | float) -> Value:
        # SOLUTION-BEGIN M08.2
        o = _lift(other)
        out = Value(self.data * o.data, (self, o), "*")

        def _backward() -> None:
            self.grad += o.data * out.grad
            o.grad += self.data * out.grad

        out._backward = _backward
        return out
        # SOLUTION-END

    def __rmul__(self, other: float) -> Value:
        # SOLUTION-BEGIN M08.2
        return self * other
        # SOLUTION-END

    def __truediv__(self, other: Value | float) -> Value:
        # SOLUTION-BEGIN M08.2
        return self * _lift(other) ** -1
        # SOLUTION-END

    def __rtruediv__(self, other: float) -> Value:
        # SOLUTION-BEGIN M08.2
        return _lift(other) * self**-1
        # SOLUTION-END

    def __neg__(self) -> Value:
        # SOLUTION-BEGIN M08.2
        return self * -1.0
        # SOLUTION-END

    def __pow__(self, k: float) -> Value:
        # SOLUTION-BEGIN M08.2
        if isinstance(k, Value) or not isinstance(k, (int, float)):
            raise TypeError("Value ** k needs a constant int or float k")
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out
        # SOLUTION-END

    def exp(self) -> Value:
        # SOLUTION-BEGIN M08.2
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
        # SOLUTION-END

    def log(self) -> Value:
        # SOLUTION-BEGIN M08.2
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out
        # SOLUTION-END

    def tanh(self) -> Value:
        # SOLUTION-BEGIN M08.2
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out
        # SOLUTION-END

    def relu(self) -> Value:
        # SOLUTION-BEGIN M08.2
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out
        # SOLUTION-END

    def backward(self) -> None:
        # SOLUTION-BEGIN M08.2
        order = toposort(self, lambda v: v._prev)
        self.grad = 1.0
        for v in order:
            v._backward()
        # SOLUTION-END


def _lift(x: Value | float) -> Value:
    """A Value as is; a number as a constant leaf."""
    # SOLUTION-BEGIN M08.2
    return x if isinstance(x, Value) else Value(x)
    # SOLUTION-END
