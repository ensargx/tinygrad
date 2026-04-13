from __future__ import annotations
import math

class Number:
    def __init__(self, val, parents: tuple[Number, ...] = (), *, requires_grad = True):
        self.val = val
        self._backprop = lambda: None
        self.parents = parents
        self._grad = 0
        self.requires_grad = requires_grad

    def __repr__(self):
        return f'Number(val={self.val}, grad={self._grad})'

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

    def __sub__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        val = self.val - other.val
        req_grad = self.requires_grad or other.requires_grad
        new = Number(val, (self, other), requires_grad=req_grad)

        def _backprop():
            self._grad += new._grad
            other._grad -= new._grad
        new._backprop = _backprop

        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Number(other) - self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Number) else Number(other)
        val = self.val * other.val
        req_grad = self.requires_grad or other.requires_grad
        new = Number(val, (self, other), requires_grad=req_grad)

        def _backprop():
            self._grad += other.val * new._grad
            other._grad += self.val * new._grad
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
            # n * x^(n-1)
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
    def zero_grad(node):
        node._grad = 0

    @staticmethod
    def topo_apply(root, apply, *args, **kwargs):
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

def main():
    x = Number(0)

    learning_rate = 0.1
    iterations = 50

    print(f"{'İter':<5} | {'e (Loss)':<10} | {'a.val':<10} | {'b.val':<10} | {'a.grad':<10}")
    print("-" * 55)

    for i in range(iterations):
        loss = x**2 - (x * 6) + 5

        Number.topo_apply(loss, Number.zero_grad)

        loss.backprop()

        Number.topo_apply(loss, Number.update_params, lr=learning_rate)

        if i % 2 == 0:
            print(f"{i:<5} | {x.val:>8.4f} | {loss.val:>12.4f} | {x._grad:>8.4f}")

    print("-" * 55)
    print(f"Final Sonuç -> x: {x.val:.4f} (Hedef: 3.0)")
    print(f"Minimum Değer -> f(x): {loss.val:.4f} (Hedef: -4.0)")

if __name__ == "__main__":
    main()
