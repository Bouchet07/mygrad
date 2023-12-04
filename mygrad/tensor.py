import numpy as np
from typing import Callable, NamedTuple, List, Optional, Union

Arraylike = Union[np.ndarray, list, float]

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    
    def __init__(self, data: Arraylike,
                 requires_grad: bool = False,
                 depends_on: Optional[List[Dependency]] = None) -> None:
        
        self.data = np.asarray(data) # Could be np.asanyarray
        self.requires_grad = requires_grad
        
        if depends_on is None: self.depends_on = []
        else: self.depends_on = depends_on
        
        self.shape = self.data.shape
        
        self.grad: Optional[np.ndarray] = None
        if requires_grad: self.grad = np.zeros_like(self.data)
    
    def __repr__(self) -> str:
        formatted_data = repr(self.data).replace('\n', '\n       ') # same number of ' ' than letters in Tensor( = 7
        return f"Tensor({formatted_data}, requires_grad={self.requires_grad})"
    
    def __array__(self) -> np.ndarray:
        return self.data
    
    def __add__(self, other: Union[Arraylike, 'Tensor']) -> 'Tensor':
        return add(self, other)
    
    def __mul__(self, other: Union[Arraylike, 'Tensor']) -> 'Tensor':
        return mul(self, other)
    
    def zero_grad(self) -> None: pass
    
    def backward(self, grad: Optional[Arraylike] = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"
        
        if grad is None: grad = np.ones_like(self.data)
        if self.grad is None: self.grad = np.zeros_like(self.data) # Line to let mypy be happy
        self.grad = self.grad + grad
        
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(self.grad)
            dependency.tensor.backward(backward_grad)
    
    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        return tensor_sum(self, axis)


def tensor_sum(t: Tensor, axis: Optional[int] = None) -> Tensor:
    data = t.data.sum(axis=axis)
    requires_grad = t.requires_grad
    
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)
    
        depends_on = [Dependency(t, grad_fn)]
    
    return Tensor(data, requires_grad, depends_on)

def add(t1: Union[Tensor, Arraylike], t2: Union[Tensor, Arraylike]):
    if not isinstance(t1, Tensor): t1 = Tensor(t1)
    if not isinstance(t2, Tensor): t2 = Tensor(t2)
    
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray, t: Tensor):
            # Sum out added dims
            ndims_added = grad.ndim - t.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
                    
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray): return grad_fn(grad, t1)
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray): return grad_fn(grad, t2)
        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(data, requires_grad, depends_on)
    
         
def mul(t1: Union[Tensor, Arraylike], t2: Union[Tensor, Arraylike]):
    if not isinstance(t1, Tensor): t1 = Tensor(t1)
    if not isinstance(t2, Tensor): t2 = Tensor(t2)
    
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad 
    
    depends_on = []
    if requires_grad:
        def grad_fn(grad: np.ndarray, t: Tensor, t_other: Tensor):
            grad = grad * t_other.data
            
            # Sum out added dims
            ndims_added = grad.ndim - t.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad
                    
    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray): return grad_fn(grad, t1, t2)
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray): return grad_fn(grad, t2, t1)
        depends_on.append(Dependency(t2, grad_fn2))
    
    return Tensor(data, requires_grad, depends_on)