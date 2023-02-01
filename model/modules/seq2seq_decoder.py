""" Defines the decoder lstm architecture for sequence prediction """

import torch
import torch.nn as nn

class seqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_len, mlp_size , num_layers=1):
        super(seqLSTM, self).__init__()
    
        """ ConvLSTM Stack definition
        Args:
            input_dim (int): size of the input 
            hidden_dim (int): size of the hidden dimention
            num_layers (int): Number of LSTM layers stacked on each other (+++multilayer bug fix -->TODO+++)
            pred_len (int): length of prediction in frames
            mlp_size (int): size of the linear projection layer
            batch_first: Whether or not dimension 0 is the batch or not
        Input:
            A tensor of size B, 2, C_vector
        Output:
            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
        """
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.mlp_size = mlp_size
        
        odometry_data_size = 3
        
        # stacking multilayer seqLSTMs over each other
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(nn.LSTMCell(input_size=cur_input_dim,
                                         hidden_size=self.hidden_dim[i]))
        self.cell_list = nn.ModuleList(cell_list)
        
        # linear projection for hidden state
        self.mlp_hidden = torch.nn.Linear(self.hidden_dim[0], self.mlp_size)
        # linear projection for odometry state
        self.mlp_odometry = torch.nn.Linear(odometry_data_size, self.mlp_size)
        self.activation = torch.nn.ReLU()

        
    def forward(self, input_h, odometry_input):
        """
        input_tensor: 
            input_h: 3-D Tensor of shape (B, 2, context_vector_size), which includes 
                both the hidden state and the cell state
            odometry_input: 3D Tensor of shape (T, B ,(Vx, Vy, Theta)), for odometry readings
        Returns:
            last_state_list, layer_output
        """
        b = input_h[0].size()[0]
            
        layer_output_list = []
        last_state_list = []
        
        self.odometry = odometry_input
        
        cur_layer_input = self._init_input(b, self.input_dim)
        hidden_state = self._extend_hidden_state(input_h)
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(self.pred_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[layer_idx], (h, c) )
                output_inner.append(h)
                # linear projections with ReLU activations applied for domain transfer
                h_projected = self.activation(self.mlp_hidden(h.view(b, -1)))
                o_projected = self.activation(self.mlp_odometry(self.odometry[t,:,:].view(b, -1)))
                cur_layer_input[layer_idx] = torch.cat((h_projected, o_projected), dim=1)
                
            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output 
                
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        last_out = layer_output_list[-1]
        last_state = last_state_list[-1]
                       
        return last_out, last_state
        
    def _init_input(self, batch_size, input_dim):
        init_inputs = []
        for i in range(self.num_layers):
            init_in = torch.zeros(batch_size, input_dim) 
            init_in = init_in.cuda() #  ++++fix cuda bug fix
            init_inputs.append(init_in)
        return init_inputs
    
    def _extend_hidden_state(self, hidden_state):
        if not isinstance(hidden_state, list):
            hidden_state = [hidden_state] * self.num_layers
        return hidden_state
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    