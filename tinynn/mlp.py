from numbers import Number
import random
from tinynn.layer import Layer

class MLP:
    def __init__(self, layer_sizes: list[int], act_fn, seed: int = 24):
        self.rng = random.Random(seed)
        self.layers = [Layer(nin, nout, act_fn=act_fn, rng=self.rng) for nin, nout in zip(layer_sizes, layer_sizes[1:])]

    def __call__(self, x) -> list[Number]:
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]
