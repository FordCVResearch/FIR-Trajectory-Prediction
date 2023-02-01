import torch

from model.modules.trajectory_encoder import TrajectoryEncoder

torch.manual_seed(1)
inputs = [torch.randn(2, 1, 4) for _ in range(30)]
inputs = torch.cat(inputs).view(len(inputs), 2, -1)

model = TrajectoryEncoder(4, 4, 1, embeded_encoding=True, in_kernel_size=4, use_gru=False)
h, c = model.forward(inputs)
    
assert (h.shape[0], h.shape[1]) == (2 , 128), "Failed"

print("Success")