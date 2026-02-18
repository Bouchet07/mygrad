from mygrad.tensor import *
from mygrad.act_fun import *
import mygrad

class Layer():

    def __init__(self, nin: int, nout: int,
                 actf: str | Callable[[Arraylike], Tensor] | None = 'relu'):
        W = np.random.randn(nin,nout)
        b = np.random.randn(1, nout)
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)
        self.actf = lambda x: None
        if actf is not None: 
            if isinstance(actf, str): 
                if actf in mygrad.act_fun.__all__: self.actf = eval(actf)
                else: raise NotImplementedError(f"{actf} not implemented yet")
            elif callable(actf): self.actf = actf
            else: raise TypeError(f"type: {type(actf)} not allowed")
    
    def __call__(self, X: Arraylike):
        return self.actf(matmul(X, self.W) + self.b)
                
    
class MLP():
    def __init__(self, nin: int, nouts: list[int], actf_l: str | list[str]) -> None:
        sz = [nin] + nouts
        if isinstance(actf_l, str): actf_l = [actf_l]*(len(nouts)-1)+[None]
        self.layers = [Layer(sz[i], sz[i+1], actf_l[i]) for i in range(len(nouts))]
    
    def __call__(self, X: Arraylike):
        for layer in self.layers:
            X = layer(X)
        return X

    def zero_grad(self):
        for layer in self.layers:
            layer.W.zero_grad()
            layer.b.zero_grad()
    
    def optimize(self, lr: float):
        for layer in self.layers:
            layer.W.data = layer.W.data - layer.W.grad * lr
            layer.b.data = layer.b.data - layer.b.grad * lr