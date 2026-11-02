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
        outputs = []
        for x_t in x:
            #reshape input to column vector
            x_t = x_t.reshape(-1, 1)
            #concatenate previous hidden state and current input
            gate_input = torch.cat((initial_hidden_state, x_t),dim=0)
            #forget gate
            f_t = torch.sigmoid(self.Wf @ gate_input + self.bf)
            #input gate
            i_t = torch.sigmoid(self.Wi @ gate_input + self.bi)
            #candidate cell state
            c_candidate = torch.tanh(self.Wc @ gate_input + self.bc)
            #update cell state
            initial_cell_state = f_t * initial_cell_state + i_t * c_candidate
            #output gate
            o_t = torch.sigmoid(self.Wo @ gate_input + self.bo)
            #update hidden state
            initial_hidden_state = o_t * torch.tanh(initial_cell_state)
            outputs.append(initial_hidden_state)
        
        outputs = torch.stack(outputs, dim=0)

        return outputs, initial_hidden_state, initial_cell_state
        
