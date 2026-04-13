import random
from tinynn.layer import Layer
from tinynn.number import Number

class MLP:
    def __init__(self, layer_sizes: list[int], act_fn, seed: int = 24):
        self.rng = random.Random(seed)
        self.layers = [Layer(nin, nout, act_fn=act_fn, rng=self.rng) for nin, nout in zip(layer_sizes, layer_sizes[1:])]

    def __call__(self, x: list[Number]) -> list[Number] | Number:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]
