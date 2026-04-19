from tinygrad.neuron import Neuron
from tinygrad.number import Number

class Layer:
    def __init__(self, nin, nout, *, act_fn, rng):
        self.rng = rng
        self.act_fn = act_fn
        self.neurons = [Neuron(nin, act_fn=act_fn, rng=rng) for _ in range(nout)]

    def __call__(self, x: list[Number]) -> list[Number]:
        outs: list[Number] = [n(x) for n in self.neurons]
        return outs

    @property
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters]
