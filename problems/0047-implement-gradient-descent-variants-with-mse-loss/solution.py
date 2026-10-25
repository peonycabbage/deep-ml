import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                    learning_rate: float, n_epochs: int, 
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights as a tensor
    """
    # Your implementation here
    weights = weights.detach().clone().requires_grad_(True)

    if method == "batch":
        for _ in range(n_epochs):
            pred = X @ weights
            loss = ((pred - y) ** 2).mean()
            loss.backward()

            with torch.no_grad():
                weights -= learning_rate*weights.grad

            weights.grad.zero_()
        return weights.detach().reshape(-1)
    elif method == "stochastic":
        for _ in range(n_epochs):
            for i in range(len(X)):
                x_i = X[i]
                y_i = y[i]

                pred_i = x_i @ weights
                loss = (pred_i - y_i) **2
                loss.backward()

                with torch.no_grad():
                    weights -= learning_rate*weights.grad
                weights.grad.zero_()
        return weights.detach().reshape(-1)

    elif method == "mini_batch":
        for _ in range(n_epochs):
            for i in range(0, len(X), batch_size):   
                x_i = X[i:i+batch_size]
                y_i = y[i:i+batch_size]

                pred_i = x_i @ weights
                loss = ((pred_i - y_i) ** 2).mean()
                loss.backward()

                with torch.no_grad():
                    weights -=learning_rate*weights.grad
                weights.grad.zero_()
        return weights.detach().reshape(-1)



