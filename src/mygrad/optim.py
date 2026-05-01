
from mygrad.tensor import Tensor

class SGD:
    def __init__(self, parameters: list[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            if p.requires_grad:
                p.zero_grad() 
                
    def step(self):
        for p in self.parameters:
            if p.requires_grad:
                # The actual weight update
                p.data = p.data - self.lr * p.grad