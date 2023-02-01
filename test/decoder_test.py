import torch
import os

from model.decoder import Decoder

seqLSTM_config = {'input_dim': 128,
               'hidden_dim': 128,
               'mlp_size': 64}

torch.manual_seed(1)
inputs_h = torch.randn(4, 128) 
inputs_c = torch.randn(4, 128) 
odometry_input = torch.randn(30, 4, 3) 

model = Decoder(seqLSTM_config, 
                prediction_len=30)
out = model.forward((inputs_h, inputs_c), odometry_input)
    
assert (out.shape[0], out.shape[1], out.shape[2]) == (30, 4, 4), "Failed"

print("Success")