import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    matrix_t = torch.as_tensor(matrix, dtype=float)
    eigenvalues = torch.linalg.eigvals(matrix_t).real

    return torch.sort(eigenvalues, descending=True).values

