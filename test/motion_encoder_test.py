import torch
import os

from model.modules.motion_encoder import MotionEncoder
from model.utils import Params

torch.manual_seed(1)
inputs = [torch.randn(4, 2, 10, 10) for _ in range(30)]
inputs = torch.cat(inputs).view(len(inputs), 4, 2, 10, 10)

json_path = os.path.join('./experiments', 'net_params.json')
params = Params(json_path)

model = MotionEncoder((10, 10), 
                      [16, 8],
                      (params.kernel_size, params.kernel_size),
                      256,
                      padding=params.padding,
                      num_layers=2)
h_out, c_out = model.forward(inputs)
    
assert (h_out.shape[0], h_out.shape[1]) == (4 , 256), "Failed"

print("Success")