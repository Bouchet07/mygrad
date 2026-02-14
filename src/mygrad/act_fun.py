import numpy as np

from mygrad.tensor import Tensor, Dependency, Tensorlike

__all__ = ['tanh', 'sigm', 'relu']

def tanh(t: Tensorlike) -> Tensor:
    if not isinstance(t, Tensor): t = Tensor(t)
    data = np.tanh(t.data)
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def sigm(t: Tensorlike) -> Tensor:
    if not isinstance(t, Tensor): t = Tensor(t)
    data = 1 / (1+np.exp(-t.data))
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * data * (1 - data)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def relu(t: Tensorlike) -> Tensor:
    if not isinstance(t, Tensor): t = Tensor(t)
    cond = t.data > 0
    data = t.data * cond
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * cond

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)