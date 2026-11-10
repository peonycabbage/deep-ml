import torch
import torch.nn.functional as F

def prelu(x: float, alpha: float = 0.25) -> torch.Tensor:
    """
    Implements the PReLU (Parametric ReLU) activation function using PyTorch.

    Args:
        x: Input value
        alpha: Slope parameter for negative values (default: 0.25)

    Returns:
        torch.Tensor: PReLU activation value
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    return (F.prelu(x, alpha))
    