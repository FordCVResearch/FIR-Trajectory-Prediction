""" Defines the encoder architecture for the the motion input """

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.utility_layers import  ConvLSTM, ConvGRU

logger = logging.getLogger(__name__)

class MotionEncoder(nn.Module):
    
    """A single layer of the motion encoder.
        Args:
        map_size (tuple): size of the input optical flow map
        num_filters (int): number of filters in ConvLSTM
        kernel_size (tuple): size of the kernel in ConvLSTM 
        output_size (int): size of the output FC layer
        num_layers (int): number of recurrent stacks (either GRU or LSTM)
        use_gru (bool): whether to use GRU or LSTM
        use_dopout_lstm (bool): whether to use Dropout layer on the outputs of 
            each LSTM layer except the last layer
        use_LeakyRelu (bool): whether to use LeakyReLU or not
    """
       
    def __init__(self, 
                 map_size,
                 num_filters, 
                 convLSTM_kernel_size,
                 output_size,
                 num_layers=1, 
                 padding=0,
                 stride=1,
                 use_gru=False,
                 use_dopout_lstm= False,
                 use_LeakyRelu= False,
                 use_MLP=True):     
        super(MotionEncoder, self).__init__()  
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        
        self.image_size = list(map_size)
        self.num_filters = num_filters
        self.kernel_size = convLSTM_kernel_size 
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_gru = use_gru
        self.use_dopout_lstm = use_dopout_lstm
        self.use_LeakyRelu = use_LeakyRelu
        self.use_MLP = use_MLP
                   
        # ConvGRU stack
        if self.use_gru:
            self.enc_rnn = ConvGRU(2, self.num_filters, self.kernel_size, 
                                    num_layers =self.num_layers)
        # ConvLSTM stack
        else:
            self.enc_rnn = ConvLSTM(2, self.num_filters, self.kernel_size, 
                                    num_layers =self.num_layers, stride=stride, padding=padding)            
        # output embedding layer   
        if self.use_MLP:
            fc_in_size = self.image_size[0] * self.image_size[1] * self.num_filters[-1]
            self.fc_out = torch.nn.Linear(fc_in_size ,self.output_size)    
        # activation
        if use_LeakyRelu:
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            self.activation = torch.nn.ReLU()
        
        
    def forward(self, x_in):
        
        """contextual encoder layer definition.
        Args:
            x_in (FloatTensor): `[T (query), #B , map_size=(2, H, W)]`
        Returns:
            x_out (FloatTensor): `[#B, output_size]`
            
        [T,#B,map_size]
                | 
                V
            ConvLSTM Stack
        [#B,convLSTM_kernel_size,(feature_map_w, feature_map_h)] ##same padding is used to preserve the size 
                |
                V
              Flatten
        [#B, convLSTM_kernel_size * feature_map_w * feature_map_h]   
                |
                V
                FC
        [#B, output_size]
        """                          
        # ConvLSTM layer
        _, self.last_states = self.enc_rnn.forward(x_in)
        self.last_h = self.last_states[0]
        self.last_c = self.last_states[1]
            
        # Flatten the lstm output
        self.h_flat = torch.flatten(self.last_h, start_dim=1)
        self.c_flat = torch.flatten(self.last_c, start_dim=1)
        
        assert self.h_flat.shape[-1] == self.c_flat.shape[-1], "last layer hidden sizes missmatch the cell state sizes"
        
        if self.use_MLP:
            # transform to encoded vector
            self.h_out = self.fc_out(self.h_flat)
            self.h_out = self.activation(self.h_out)
            
            self.c_out = self.fc_out(self.c_flat)
            self.c_out = self.activation(self.c_out)
        else:
            self.h_out = self.h_flat
            self.c_out = self.c_flat
            
        return (self.h_out, self.c_out)
        
        
    
        
        
        
    