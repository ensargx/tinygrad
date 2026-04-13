from random import Random
from tinynn.number import Number

class Neuron:
    def __init__(self, nin, *, act_fn, rng: Random):
        self.w = [Number(rng.uniform(-1,1)) for _ in range(nin)]
        self.b = Number(rng.uniform(-1,1))
        self.act_fn = act_fn

    def __call__(self, x) -> Number:
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return self.act_fn(act) if self.act_fn else act

    @property
    def parameters(self):
        return self.w + [self.b]
