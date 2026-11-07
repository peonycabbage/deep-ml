import torch

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)

    
    # Your code here: use autograd to compute gradients
    for _ in range(iterations):
        y_hat = X_t @ theta
        error = y_hat - y_t
        loss = torch.mean(error ** 2) / 2
        loss.backward()
        with torch.no_grad():
            theta -= alpha * theta.grad
        theta.grad.zero_()
    return theta.flatten().detach()

    
    
    
    
    
    
    
    
    
    
    # for _ in range(iterations):
    #     y_hat = X_t @ theta
    #     errors = y_hat - y_t
    #     loss = torch.mean(errors ** 2)
    #     loss.backward()
    #     with torch.no_grad():
    #         theta -= alpha * theta.grad
    #     theta.grad.zero_()
    # return theta.flatten().detach()
