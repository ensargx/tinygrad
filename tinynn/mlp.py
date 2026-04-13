from tinynn.layer import Layer
from tinynn.number import Number

class MLP:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)

    def __call__(self, x: list[Number]) -> list[Number] | Number:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    @property
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters]
