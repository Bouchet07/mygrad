from numbers import Real
from math import log

__all__ = ['Value']

def _safe_log(x):
    return log(x) if x > 0 else float('-inf')

def _build_topo(node, topo = None, visited=None):
    if topo is None: topo = []
    if visited is None: visited = set()
    if node not in visited:
        visited.add(node)
        for child in node.children:
            _build_topo(child, topo, visited)
        topo.append(node)
    return topo

class Value:
    def __init__(self, x: Real, _children = (), _op = None):
        self.data = x
        self.grad = 0
        self.children = set(_children)
        self.op = _op
        self._backward = lambda: None
    
    def __repr__(self):
        return f'Value(data={self.data:.4f}, grad={self.grad:.4f})'
    
    def __add__(self, other):
        if isinstance(other, Real): other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        if isinstance(other, Real): other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, other):
        if isinstance(other, Real): other = Value(other)
        out = Value(self.data ** other.data, (self, other), '^')
        
        def _backward():
            self.grad += other.data * self.data ** (other.data-1) * out.grad
            other.grad += self.data ** other.data / _safe_log(self.data)
        out._backward = _backward
        
        return out
    
    def relu(self):
        cond = self.data > 0
        out = Value(self.data * cond, (self,), 'relu')
        def _backward():
            self.grad += cond * out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        self.grad = 1
        for node in reversed(_build_topo(self)):
            node._backward()
    
    def __neg__(self):
        return self * (-1)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rpow__(self, other):
        return Value(other) ** self
    
    def __truediv__(self, other):
        return self * other**(-1)
    
    def __rtruediv__(self, other):
        return self**(-1) * other
    