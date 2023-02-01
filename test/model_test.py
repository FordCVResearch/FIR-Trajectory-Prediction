import torch
import os

from model.network.base_model import Encoder_Decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traj_config = {'input_size': 4,
               'hidden_size': 4,
               'num_layers': 1,
               'embeded_encoding': False,
               'out_kernel_size': 256}

contex_config = {'image_size': (3, 224, 224),
               'num_filters': [32, 16],
               'convLSTM_kernel_size': (1, 1),
               'output_size':1024,
               'num_layers':2,
               'cnn_model': None,
               'use_MLP': False}

motion_config = {'map_size': (10, 10),
               'num_filters': [16, 8],
               'convLSTM_kernel_size': (1, 1),
               'output_size': 512,
               'num_layers':2,
               'use_MLP': False}

encoder_config = {'traj_config': traj_config,
                  'contex_config': contex_config,
                  'motion_config' :motion_config,
                  'use_mlp': True,
                  'mlp_kernel_size': 1024}

decoder_config = {'input_dim': 512, #must be 2x the mlp_size
               'hidden_dim':  1024,
               'mlp_size': 256}
torch.manual_seed(1)

boxes = [torch.randn(2, 4) for _ in range(30)]
boxes = torch.cat(boxes).view(len(boxes), 2, -1)

images = [torch.randn(2, 224, 224, 3) for _ in range(30)]
images = torch.cat(images).view(len(images), 2, 3, 224,224)

motions = [torch.randn(2, 2, 640, 480) for _ in range(30)]
motions = torch.cat(motions).view(len(motions), 2, 2, 640, 480)

odometry = torch.randn(10, 2, 3) 

model = Encoder_Decoder(encoder_config, decoder_config, prediction_length=10, roi_size=(10,10), image_size=(224,224))
pred_box = model.forward(boxes, images, motions, odometry)

assert (pred_box.shape[0], pred_box.shape[1], pred_box.shape[2]) == (10, 2, 4), "Failed"
print("Success")