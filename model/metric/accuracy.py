"""Defines the accuracy metric"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy_fn(bb_predic, bb_gt):
    """mean absolute error (MAE) L1 norm definition.
    Args:
        bb_pred (FloatTensor): `[T (query), #B , bb_size=4]`
        bb_gt (FloatTensor): `[T (query), #B , bb_size=4]`
    """
    assert bb_predic.shape == bb_gt.shape, "inconsistant tensor shape"
    T = bb_predic.shape[0]
    
    total_acc_fn = nn.L1Loss(reduction='mean')
    seq_acc_fc = nn.L1Loss(reduction='none')
    total_acc = total_acc_fn(bb_predic, bb_gt)
    
    seq_acc_list = []
    for t in range(T):
        seq_acc_list.append(seq_acc_fc(bb_predic[t,:,:], bb_gt[t,:,:]))
        
    seq_acc = torch.stack(seq_acc_list, dim=0)
    
    return total_acc, seq_acc