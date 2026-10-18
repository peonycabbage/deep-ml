import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your code here

    a = m[0,0]
    b = m[0,1]
    c = m[1,0]
    d = m[1,1]
    det = a * d - b * c
    if det == 0:
        return None
    
    
    rearranged = torch.stack([torch.stack([d, -b]), torch.stack([-c, a])])

    return rearranged / det

