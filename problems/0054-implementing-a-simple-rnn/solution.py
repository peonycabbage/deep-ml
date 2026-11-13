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
    hidden_state = torch.as_tensor(initial_hidden_state, dtype=torch.float32)
    weight_ih = torch.as_tensor(Wx, dtype=torch.float32)
    weight_hh = torch.as_tensor(Wh, dtype=torch.float32)
    bias = torch.as_tensor(b, dtype=torch.float32)

    for input_t in input_sequence:
        input_projection = torch.matmul(weight_ih, input_t)
        hidden_projection = torch.matmul(weight_hh, hidden_state) 
        hidden_state = torch.tanh(input_projection + hidden_projection + bias)
    return torch.round(hidden_state, decimals=4)
 