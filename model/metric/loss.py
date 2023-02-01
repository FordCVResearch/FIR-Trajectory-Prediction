"""Defines the losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(bb_predic, bb_gt):
    """squared L2 norm definition.
    Args:
        bb_pred (FloatTensor): `[T (query), #B , bb_size=4]`
        bb_gt (FloatTensor): `[T (query), #B , bb_size=4]`
    """
    assert bb_predic.shape == bb_gt.shape, "inconsistant tensor shape"
    T = bb_predic.shape[0]
    
    loss = nn.MSELoss()
    seq_loss_list = []
    for t in range(T):
        seq_loss_list.append(loss(bb_predic[t,:,:], bb_gt[t,:,:]))
    
    return sum(seq_loss_list) / len(seq_loss_list)