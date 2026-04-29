import numpy as np
from mygrad.tensor import Tensor, as_tensor, Dependency, Tensorlike
from typing import Callable

import sys

__all__ = ['tanh', 'sigm', 'relu', 'lrelu', 'elu', 'selu', 'gelu','identity', 'step']

def _apply_act(t: Tensor, data: np.ndarray, grad_logic: Callable[[], np.ndarray], op_name: str | None = None) -> Tensor:
    """
    Internal helper. Assumes 't' is already a Tensor object.
    """
    depends_on = []
    if t.requires_grad:
        # The closure captures the math logic specific to the activation
        depends_on.append(Dependency(t, lambda grad: grad * grad_logic()))
   
    out = Tensor(data, requires_grad=t.requires_grad, depends_on=depends_on)
    out._op = op_name if op_name is not None else "UnknownAct"
    return out

# --- Public Functions ---

def identity(t: Tensorlike) -> Tensor:
    """Identity function: f(x) = x."""
    t = as_tensor(t)
    return _apply_act(t, t.data, lambda: np.ones_like(t.data), "Identity")

def step(t: Tensorlike) -> Tensor:
    """Step function: f(x) = 1 if x > 0 else 0."""
    t = as_tensor(t)
    data = (t.data > 0).astype(float)
    return _apply_act(t, data, lambda: np.zeros_like(t.data), "Step")  # Derivative is zero almost everywhere

def relu(t: Tensorlike) -> Tensor:
    """Rectified Linear Unit: max(0, x)."""
    t = as_tensor(t)  # Convert once here
    mask = (t.data > 0).astype(t.data.dtype)
    return _apply_act(t, t.data * mask, lambda: mask, "ReLU")  # Derivative is 1 where mask is True, else 0

def tanh(t: Tensorlike) -> Tensor:
    """Hyperbolic tangent activation function."""
    t = as_tensor(t)
    data = np.tanh(t.data)
    return _apply_act(t, data, lambda: 1 - data**2, "Tanh")

def sigm(t: Tensorlike) -> Tensor:
    """Sigmoid activation function."""
    t = as_tensor(t)
    data = 1 / (1 + np.exp(-t.data))
    return _apply_act(t, data, lambda: data * (1 - data), "Sigmoid")

def lrelu(t: Tensorlike, alpha: float = 0.01) -> Tensor:
    """Leaky ReLU: x if x > 0 else alpha * x."""
    t = as_tensor(t)
    mask = (t.data > 0).astype(t.data.dtype)
    data = np.where(mask, t.data, alpha * t.data)
    return _apply_act(t, data, lambda: np.where(mask, 1.0, alpha), "LeakyReLU")

def elu(t: Tensorlike, alpha: float = 1.0) -> Tensor:
    """Exponential Linear Unit."""
    t = as_tensor(t)
    mask = (t.data > 0).astype(t.data.dtype)
    data = np.where(mask, t.data, alpha * (np.exp(t.data) - 1))
    return _apply_act(t, data, lambda: np.where(mask, 1.0, data + alpha), "ELU")

def selu(t: Tensorlike, 
         alpha: float = 1.6732632423543772, 
         scale: float = 1.0507009873554805) -> Tensor:
    """Scaled Exponential Linear Unit."""
    t = as_tensor(t)
    mask = (t.data > 0).astype(t.data.dtype)
    data = np.where(mask, scale * t.data, scale * alpha * (np.exp(t.data) - 1))
    # Note: data + scale * alpha is a common optimization for the SELU derivative
    return _apply_act(t, data, lambda: np.where(mask, scale, data + scale * alpha), "SELU")

def gelu(t: Tensorlike) -> Tensor:
    """Gaussian Error Linear Unit."""
    t = as_tensor(t)
    x = t.data
    
    # Precompute the expensive inner terms once
    inner_poly = x + 0.044715 * x**3
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    tanh_term = np.tanh(sqrt_2_over_pi * inner_poly)
    
    # Forward pass
    data = 0.5 * x * (1 + tanh_term)
    
    # Precompute derivative components so the lambda is fast
    # d(tanh)/dx = 1 - tanh^2
    sech_squared = 1 - tanh_term**2
    d_inner = sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)

    return _apply_act(t, data, lambda: 0.5 * (1 + tanh_term) + 0.5 * x * sech_squared * d_inner, "GELU")