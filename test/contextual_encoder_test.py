import torch
import os

from model.modules.contextual_encoder import ContextEncoder
from model.utils import Params

CNN_MODELS =['resnet', 'vgg', 'inception']
json_path = os.path.join('./experiments', 'net_params.json')
params = Params(json_path)

torch.manual_seed(1)
inputs = [torch.randn(2, 224, 224, 3) for _ in range(30)]
inputs = torch.cat(inputs).view(len(inputs), 2,3, 224,224)

model = ContextEncoder((3, 224, 224),
                        [32,16], 
                       (params.kernel_size, params.kernel_size),
                       256,
                    #    cnn_model=CNN_MODELS[1],
                       padding=params.padding,
                       num_layers=2,
                       use_MLP=False)
h_out, c_out = model.forward(inputs)
    
assert (h_out.shape[0], h_out.shape[1]) == (2, 256), "Failed"

print("Success")