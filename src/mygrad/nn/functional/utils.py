from mygrad.tensor import Tensor, Dependency
from typing import Callable

import numpy as np

def _apply_fun(t: Tensor, data: np.ndarray, grad_logic: Callable[[], np.ndarray], op_name: str | None = None) -> Tensor:
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