import torch

def confusion_matrix(data: list) -> torch.Tensor:
    """
    Generate a 2x2 confusion matrix for binary classification.

    Args:
        data: A list of [y_true, y_pred] pairs for binary labels (0 or 1)

    Returns:
        A 2x2 torch.Tensor confusion matrix arranged as [[TP, FN], [FP, TN]]
    """
    data = torch.as_tensor(data, dtype=torch.float32)

    y_true = data[:, 0]
    y_pred = data[:, 1]

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred ==0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred ==0)).sum()


    return torch.stack([torch.stack([tp, fn]), torch.stack([fp, tn])])