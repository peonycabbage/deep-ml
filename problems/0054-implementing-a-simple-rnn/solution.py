import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    input_sequence = torch.as_tensor(input_sequence, dtype=torch.float32)
    ht = torch.as_tensor(initial_hidden_state, dtype=torch.float32)
    Wx = torch.as_tensor(Wx, dtype=torch.float32)
    Wh = torch.as_tensor(Wh, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)


    for x in input_sequence:

        ht = torch.matmul(Wx, x) + torch.matmul(Wh, ht) + b
        ht = torch.tanh(ht)
    return torch.round(ht, decimals=4)