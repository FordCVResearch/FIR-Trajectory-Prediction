import os
import datetime
import time

from tqdm import trange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import data_iterator
from model.utils import *

def test(model, loss_fn, metrics, params, data_iterator, log_dir, train_size=None):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    model.eval()
    
    summ = []
    num_test_steps = int(np.ceil(train_size / params.batch_size))
    
    for _ in range(num_test_steps):
        batch_images, batch_motions, batch_boxes, batch_odometery, batch_groundt = next(data_iterator)
        # batch first -> qeue first
        batch_images = batch_images.permute(1, 0, 2, 3, 4)
        batch_motions = batch_motions.permute(1, 0, 2, 3, 4)
        batch_boxes = batch_boxes.permute(1, 0, 2)
        batch_odometery = batch_odometery.permute(1, 0, 2)
        batch_groundt = batch_groundt.permute(1, 0, 2)
        
        output_batch = model(batch_boxes, batch_images, batch_motions, batch_odometery)
        loss = loss_fn(output_batch, batch_groundt)
        mae, _ = metrics(output_batch, batch_groundt)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        batch_groundt = batch_groundt.data.cpu().numpy()

        summary_batch = {metric: metrics[metric](output_batch, batch_groundt)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    return metrics_mean