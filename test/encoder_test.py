import torch
import os

from model.encoder import Encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_MODELS =['resnet', 'vgg', 'inception']

traj_config = {'input_size': 4,
               'hidden_size': 4,
               'num_layers': 1,
               'embeded_encoding': True,
               'out_kernel_size': 128}

contex_config = {'image_size': (3, 224, 224),
               'num_filters': [32, 16],
               'convLSTM_kernel_size': (1, 1),
               'output_size':256,
               'num_layers':2,
               'cnn_model': None,
               'use_MLP': True}

motion_config = {'map_size': (10, 10),
               'num_filters': [16, 8],
               'convLSTM_kernel_size': (1, 1),
               'output_size': 256,
               'num_layers':2,
               'use_MLP': True}

torch.manual_seed(1)
traj_inputs = [torch.randn(2, 4) for _ in range(30)]
traj_inputs = torch.cat(traj_inputs).view(len(traj_inputs), 2, -1)

contx_inputs = [torch.randn(2, 224, 224, 3) for _ in range(30)]
contx_inputs = torch.cat(contx_inputs).view(len(contx_inputs), 2, 3, 224,224)

motion_inputs = [torch.randn(2, 2, 640, 480) for _ in range(30)]
motion_inputs = torch.cat(motion_inputs).view(len(motion_inputs), 2, 2, 10, 10)

encoder_model = Encoder(traj_config, contex_config, motion_config)
learned_out = encoder_model.forward(traj_inputs, contx_inputs, motion_inputs)

assert (learned_out.shape[0], learned_out.shape[1]) == (2 , 256), "Failed"
print("Success")
