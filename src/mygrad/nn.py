import numpy as np

from mygrad.tensor import Tensor, as_tensor, Tensorlike

class Module:
    def parameters(self) -> list[Tensor]:
        """Returns a list of all trainable Tensors in this module."""
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            # (Optional: handle lists of modules/tensors here)
        return params

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        # Scale weights to prevent exploding/vanishing gradients
        scale = 1.0 / np.sqrt(in_features)
        W = np.random.randn(in_features, out_features) * scale
        
        # Biases are usually initialized to zero
        b = np.zeros((1, out_features))
        
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)
    
    def __call__(self, X: Tensorlike) -> Tensor:
        # Just the affine transformation
        return as_tensor(X) @ self.W + self.b

class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, X: Tensorlike) -> Tensor:
        out = as_tensor(X)
        for layer in self.layers:
            out = layer(out)
        return out
        
    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params