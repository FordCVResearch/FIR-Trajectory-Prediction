""" Defines the encoder architecture for the the trajectory input """

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class TrajectoryEncoder(nn.Module):
    
    """A single layer of the trajectory encoder.
        Args:
        input_size (int): input dimension of trajectory features
        hidden_size (int): features in the hidden state
        num_layers (int): number of recurrent stacks (either GRU or LSTM)
        use_gru (bool): whether to use GRU or LSTM
        embeded_encoding (bool): whether to use FC layer before the recurrent stacks 
        in_kernel_size (int): size of the input FC layer
        out_kernel_size (int): size of the output FC layer
        use_dopout_lstm (bool): whether to use Dropout layer on the outputs of 
            each LSTM layer except the last layer
        use_LeakyRelu (bool): whether to use LeakyReLU or not
    """
       
    def __init__(self, input_size, 
                 hidden_size, 
                 num_layers, 
                 use_gru=False,
                 embeded_encoding=True,
                 in_kernel_size= 128,
                 out_kernel_size= 128,
                 use_dopout_lstm= False,
                 use_LeakyRelu= False):   
        super(TrajectoryEncoder, self).__init__()  
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.embeded_encoding = embeded_encoding
        self.in_kernel_size = in_kernel_size
        self.out_kernel_size = out_kernel_size
        self.use_dopout_lstm = use_dopout_lstm
        self.use_LeakyRelu = use_LeakyRelu
        
        # input embedding layer
        self.fc_in = torch.nn.Linear(self.input_size, self.in_kernel_size)
            
        # GRU stack
        if self.use_gru:
            self.enc_rnn = torch.nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        # LSTM stack
        elif self.use_dopout_lstm:
            self.enc_rnn = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout= 0.5) 
        else:
            self.enc_rnn = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
            
        # output embedding layer   
        self.fc_out = torch.nn.Linear(self.hidden_size ,self.out_kernel_size)
        
        # activation
        if use_LeakyRelu:
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            self.activation = torch.nn.ReLU()
        
        
    def forward(self, x_in):
        
        """trajectory encoder layer definition.
        Args:
            x_in (FloatTensor): `[T (query), #B , input_size]`
        Returns:
            x_out (FloatTensor): `[#B, output_size]`
        
        [T,#B,input_size]
                |
                V
                FC
        [T,#B,in_kernel_size]
                |
                V
               RNN
        [1,#B,hidden_size]
                |
                V
                FC
        [#B,out_kernel_size]                            
        """
        residual = x_in  
        B= x_in.shape[1]
       
        if self.embeded_encoding:  
            x_in = self.fc_in(x_in)
            x_in = self.activation(x_in)
            
        _ , self.hidden = self.enc_rnn(x_in)
        
        if self.embeded_encoding:  
            self.h_out = self.fc_out(self.hidden[0])
            self.h_out = self.activation(self.h_out)
            
            self.c_out = self.fc_out(self.hidden[1])
            self.c_out = self.activation(self.c_out)
        else:
            self.h_out = self.hidden[0]
            self.c_out = self.hidden[1]
        
        return (self.h_out[-1], self.c_out[-1])
        
        
    
        
        
        
    