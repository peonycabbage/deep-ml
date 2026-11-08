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
    theta = weights.clone().detach().requires_grad_(True)
    for _ in range(n_epochs):
        if method == 'batch':
            y_hat = X @ theta
            error = y_hat - y
            loss = torch.mean(error ** 2)
            
            loss.backward()
            with torch.no_grad():
                theta -= learning_rate * theta.grad
            theta.grad.zero_()

        elif method == 'stochastic':
            for i in range(len(X)):
                y_hat = X[i] @ theta
                error = y_hat - y[i]
                loss = error ** 2
                loss.backward()

                with torch.no_grad():
                    theta -= learning_rate * theta.grad
                theta.grad.zero_()
        elif method == 'mini_batch':
            for i in range(0, len(X), batch_size):
                y_hat = X[i:i+batch_size] @ theta
                error = y_hat - y[i:i+batch_size]
                loss = torch.mean(error ** 2)
                loss.backward()

                with torch.no_grad():
                    theta -= learning_rate * theta.grad
                theta.grad.zero_()
    return theta.flatten().detach()
