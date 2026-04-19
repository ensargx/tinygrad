from __future__ import annotations
import math

class Number:
    def __init__(self, val, parents: tuple[Number, ...] = (), *, requires_grad = True):
        self.val = val
        self._backprop = lambda: None
        self.parents = parents
        self._grad = 0.0
        self.requires_grad = requires_grad

    def __repr__(self):
        return f'Number(val={self.val}, grad={self._grad})'

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Number(other) - self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * (other ** -1)

    def __add__(self, other):
        other = other if isinstance(other, Number) else Number(other, requires_grad=False)
        val = self.val + other.val
        req_grad = self.requires_grad or other.requires_grad
        new = Number(val, (self, other), requires_grad=req_grad)

        def _backprop():
            self._grad += new._grad
            other._grad += new._grad
        new._backprop = _backprop

        return new

    def __mul__(self, other):
        other = other if isinstance(other, Number) else Number(other, requires_grad=False)
        val = self.val * other.val
        req_grad = self.requires_grad or other.requires_grad
        new = Number(val, (self, other), requires_grad=req_grad)

        def _backprop():
            self._grad += other.val * new._grad
            other._grad += self.val * new._grad
        new._backprop = _backprop

        return new

    def exp(self):
        val = math.exp(self.val)
        new = Number(val, (self,), requires_grad=self.requires_grad)

        def _backprop():
            self._grad += new.val * new._grad
        new._backprop = _backprop

        return new

    def tanh(self):
        val = math.tanh(self.val)
        new = Number(val, (self,), requires_grad=self.requires_grad)

        def _backprop():
            self._grad += (1 - val**2) * new._grad
        new._backprop = _backprop

        return new

    def relu(self):
        val = self.val if self.val > 0 else 0
        new = Number(val, (self,), requires_grad=self.requires_grad)

        def _backprop():
            self._grad += new._grad * (1 if new.val > 0 else 0)
        new._backprop = _backprop

        return new

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only numbers"
        val = self.val ** other
        new = Number(val, (self,), requires_grad=self.requires_grad)

        def _backprop():
            self._grad += (other * (self.val ** (other - 1))) * new._grad
        new._backprop = _backprop

        return new

    def backprop(self):
        self._grad = 1
        Number.topo_apply(self, Number.apply_backprop)

    @staticmethod
    def update_params(node: Number, lr):
        if node.requires_grad:
            node.val -= lr * node._grad

    @staticmethod
    def apply_backprop(node: Number):
        node._backprop()

    @staticmethod
    def zero_grad(node: Number):
        node._grad = 0

    @staticmethod
    def topo_apply(root: Number, apply, *args, **kwargs):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(root)

        order = reversed(topo)

        for node in order:
            apply(node, *args, **kwargs)
