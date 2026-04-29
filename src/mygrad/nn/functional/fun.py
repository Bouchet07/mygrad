from mygrad.tensor import Tensor, as_tensor, Tensorlike, Dependency
import numpy as np

def cross_entropy(logits: Tensorlike, targets: Tensorlike) -> Tensor:
    logits = as_tensor(logits)
    targets_np = np.asarray(targets)
    
    if targets_np.shape != logits.shape:
        targets_np = targets_np.reshape(logits.shape)
    
    # Shift logits for numerical stability (prevent np.exp overflow)
    shifted_logits = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    
    # Calculate probabilities (Softmax)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Calculate Cross Entropy Loss
    N = logits.shape[0] # Batch size
    log_probs = np.log(probs + 1e-12) # Small epsilon to prevent log(0)
    loss_data = -np.sum(targets_np * log_probs) / N
    
    depends_on = []
    if logits.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (probs - targets_np) / N
        
        # Register the dependency in your graph
        depends_on.append(Dependency(logits, grad_fn))
        
    # Return the final scalar loss as a Tensor
    out = Tensor(loss_data, requires_grad=logits.requires_grad, depends_on=depends_on)
    out._op = "CrossEntropyLoss"
    return out

def binary_cross_entropy(probs: Tensorlike, targets: Tensorlike) -> Tensor:
    probs = as_tensor(probs)
    targets_np = np.asarray(targets)
    
    if targets_np.shape != probs.shape:
        targets_np = targets_np.reshape(probs.shape)
    
    N = probs.shape[0]
    # BCE Loss
    loss_data = -np.sum(targets_np * np.log(probs.data + 1e-12) + 
                        (1 - targets_np) * np.log(1 - probs.data + 1e-12)) / N
    
    # 2. Backward pass
    depends_on = []
    if probs.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            p = probs.data
            grad_bce = (p - targets_np) / (p * (1 - p) + 1e-12)
            return grad * grad_bce / N
            
        depends_on.append(Dependency(probs, grad_fn))
        
    out = Tensor(loss_data, requires_grad=probs.requires_grad, depends_on=depends_on)
    out._op = "BCEWithLogitsLoss"
    return out

def binary_cross_entropy_with_logits(logits: Tensorlike, targets: Tensorlike) -> Tensor:
    logits = as_tensor(logits)
    targets_np = np.asarray(targets)
    
    if targets_np.shape != logits.shape:
        targets_np = targets_np.reshape(logits.shape)
    
    # 1. Forward pass (Sigmoid + BCE)
    # Sigmoid: 1 / (1 + exp(-x))
    probs = 1 / (1 + np.exp(-logits.data))
    
    N = logits.shape[0]
    # BCE Loss
    loss_data = -np.sum(targets_np * np.log(probs + 1e-12) + 
                        (1 - targets_np) * np.log(1 - probs + 1e-12)) / N
    
    # 2. Backward pass
    depends_on = []
    if logits.requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            # The fused gradient for Sigmoid + BCE is exactly the same shape!
            return grad * (probs - targets_np) / N
            
        depends_on.append(Dependency(logits, grad_fn))
        
    out = Tensor(loss_data, requires_grad=logits.requires_grad, depends_on=depends_on)
    out._op = "BCEWithLogitsLoss"
    return out
