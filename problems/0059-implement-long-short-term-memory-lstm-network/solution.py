import torch

class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wi = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wc = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)
        self.Wo = torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float64)

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(self, x: torch.Tensor, initial_hidden_state: torch.Tensor, initial_cell_state: torch.Tensor):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        combined_tensor = 0
        c_t = initial_cell_state
        h_t = initial_hidden_state
        outputs = []
        for x_t in x:
            x_t = x_t.reshape(-1,1)
            combined_tensor = torch.cat((h_t, x_t), dim=0)
            f_t = torch.sigmoid(torch.matmul(self.Wf, combined_tensor) + self.bf)
            i_t =  torch.sigmoid(torch.matmul(self.Wi, combined_tensor)+self.bi)
            candidate_ct = torch.tanh(torch.matmul(self.Wc, combined_tensor)+self.bc)
            c_t = f_t * c_t + i_t * candidate_ct
            o_t = torch.sigmoid(torch.matmul(self.Wo, combined_tensor)+ self.bo)
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t.clone())
        outputs = torch.stack(outputs,dim=0)
        return outputs, h_t, c_t





