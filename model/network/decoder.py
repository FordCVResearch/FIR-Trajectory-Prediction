""" Defines the seq2seq decoder architecture """

import logging

import torch
import torch.nn as nn

from model.modules.seq2seq_decoder import seqLSTM

logger = logging.getLogger(__name__)

class Decoder(nn.Module):
    
    """An implementation of the seq2seq decoder.
        Args:
        traj_config (dict): input config for trajectory encoder
        prediction_len (int): length of the future sequence for prediction
    """
       
    def __init__(self, 
                 seqLSTM_config,
                 prediction_len,
                 mlp_kernel_size=4):   
        super(Decoder, self).__init__()  
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        
        self.seqLSTM_config = seqLSTM_config
        self.mlp_kernel_size = mlp_kernel_size
        self.prediction_len = prediction_len
        
        # setup the seqLSTM encoder
        self.seq_decoder = seqLSTM(self.seqLSTM_config['input_dim'], 
                                   self.seqLSTM_config['hidden_dim'], 
                                   self.prediction_len, 
                                   self.seqLSTM_config['mlp_size'])
        
        self.fc_out = torch.nn.Linear(self.seqLSTM_config['hidden_dim'], self.mlp_kernel_size)
        self.activation = torch.nn.ReLU()
        
    def forward(self, encoded_in, odometry_in):
        
        bb_list = []
        inputs_h, inputs_c = encoded_in
        out, _ = self.seq_decoder.forward((inputs_h, inputs_c), odometry_in)
        
        for t in range(self.prediction_len):
            bb_list.append(self.activation(self.fc_out(out[t,:,:])))
            
        return torch.stack(bb_list, dim=0)
