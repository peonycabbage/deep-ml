import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    input_tensor = torch.as_tensor(z, dtype=torch.float32)
    output = torch.sigmoid(input_tensor)
    return round(output.item(), 4)
