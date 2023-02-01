""" Definition of some Utility Layers including: Identity, ConvLSTM, ConvGRU """

import torch
import torch.nn as nn

class Identity(nn.Module):
    """
     Identity layer to substitute the classification layers in backbone CNN 
    """
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, stride=1, padding=1):
        """ Initialize ConvLSTM cell Parameters
        Args:
            input_dim (int): Number of channels of input tensor.
            num_filters (int): Number of filters in the hidden state.
            kernel_size (int, int): Size of the convolutional kernel.
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = num_filters
        self.kernel_size = kernel_size
       
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                        out_channels=4 * self.hidden_dim,
                        kernel_size=self.kernel_size,
                        padding=padding,
                        stride=stride)
        
    def forward(self, input_tensor, current_state):
        h_cur, c_cur = current_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)  #--> X^T.Wx + h^T-1.Wh
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # implementation of this paper https://arxiv.org/abs/1506.04214
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
           
    
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_size, num_layers=1, stride=1, padding=1,
                 batch_first=False, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        """ ConvLSTM Stack definition
        Args:
            input_dim (int): Number of channels in input
            hidden_dim (int): Number of filters in hidden channels
            kernel_size (int): Size of kernel in convolutions
            num_layers (int): Number of LSTM layers stacked on each other
            stride (int): Number of strides
            padding (int): Number of padding 
            batch_first: Whether or not dimension 0 is the batch or not
            return_all_layers: Return the list of computations for all layers
        Input:
            A tensor of size T, B, C, H, W
        Output:
            A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
                0 - layer_output_list is the list of lists of length T of each output
                1 - last_state_list is the list of last states
                        each element of the list is a tuple (h, c) for hidden state and memory
        Example:
            >> x = torch.rand((10, 32, 64, 128, 128))
            >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
            >> _, last_states = convlstm(x)
            >> h = last_states[0][0]  # 0 for layer index, 0 for h index
        """
        # Some sanity checks
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(num_filters, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        self.padding = padding
        self.stride = stride
        
        # stacking multilayer convLSTMs over each other
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim= cur_input_dim, 
                                          num_filters= self.hidden_dim[i],
                                          kernel_size= self.kernel_size[i],
                                          padding= self.padding,
                                          stride= self.stride))
        self.cell_list = nn.ModuleList(cell_list)
                                                                              
    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: 
            5-D Tensor of shape (t, b, c, h, w) 
        hidden_state: todo
            None. todo implement stateful
        Returns:
            last_state_list, layer_output
        """
        # switch to batch first
        # (t, b, c, h, w) -> (b, t, c, h, w)
        input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 current_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
                
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]
            
        return layer_output_list, last_state_list
            
                       
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
        
        
class ConvGRU(nn.Module):
    """
    ConvGRU Cell implementation
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(ConvGRU, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext