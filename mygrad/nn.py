import random
from .base import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, actf='relu'):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.actf = actf

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.actf is None: return act
        return eval(f'act.{self.actf}()')

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'linear' if self.actf is None else self.actf}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, actf='relu'):
        self.neurons = [Neuron(nin, actf) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts, actf_l='relu'):
        sz = [nin] + nouts
        if isinstance(actf_l, str): actf_l = [actf_l]*(len(nouts)-1)+[None]
        self.layers = [Layer(sz[i], sz[i+1], actf_l[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"