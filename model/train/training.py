""" training class """

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

class Train_and_Evaluate():
    """Train the model and evaluate every epoch.
    Args:
        model_spec: (dict) the encoder-decoder network
        train_data: (dict) training data
        val_data: (dict) validaion data
        log_dir: (str) log directory location
    """
    
    def __init__(self, model, loss, opt, metric, hyper_params, train_data, val_data, log_dir):
        
        self.model = model
        self.train_data =  train_data
        self.val_data = val_data
        self.log_dir = log_dir
        
        # Get relevant graph operations or nodes needed for training
        self.loss_fn = loss
        self.optimizer = opt

        self.metrics = metric
        self.params = hyper_params
        
    def train_step(self, data_iterator, num_steps):
        # set model to training mode
        self.model.train()
        # summary for current training loop and a running average object for loss
        train_summ = []
        loss_avg = RunningAverage()
        
        # Use tqdm for progress bar
        t = trange(num_steps)
        for i in t:
            batch_images, batch_motions, batch_boxes, batch_odometery, batch_groundt = next(data_iterator)
            # batch first -> qeue first
            batch_images = batch_images.permute(1, 0, 2, 3, 4)
            batch_motions = batch_motions.permute(1, 0, 2, 3, 4)
            batch_boxes = batch_boxes.permute(1, 0, 2)
            batch_odometery = batch_odometery.permute(1, 0, 2)
            batch_groundt = batch_groundt.permute(1, 0, 2)
            
            output_batch = self.model(batch_boxes, batch_images, batch_motions, batch_odometery)
            loss = self.loss_fn(output_batch, batch_groundt)
            mae, _ = self.metrics(output_batch, batch_groundt)
            # clear previous gradients, compute gradients of all variables wrt loss
            self.optimizer.zero_grad()
            loss.backward()
            # performs updates using calculated gradients
            self.optimizer.step()
            # Evaluate summaries only once in a while
            if i % self.params.save_summary_steps == 0:
                summary_batch ={
                    'loss' : loss.item(),
                    'mae' : mae.item()}
                train_summ.append(summary_batch)
                
            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        
        # compute mean of all metrics in summary
        self.metrics_mean = {metric: np.mean([x[metric] for x in train_summ]) for metric in train_summ[0]}    
        
     
    def valid_step(self, data_iterator, num_steps):
        # set model to evaluation mode
        self.model.eval()
        # summary for current eval loop
        val_summ = []
        for _ in range(num_steps):
            batch_images, batch_motions, batch_boxes, batch_odometery, batch_groundt = next(data_iterator)
            # batch first -> qeue first
            batch_images = batch_images.permute(1, 0, 2, 3, 4)
            batch_motions = batch_motions.permute(1, 0, 2, 3, 4)
            batch_boxes = batch_boxes.permute(1, 0, 2)
            batch_odometery = batch_odometery.permute(1, 0, 2)
            batch_groundt = batch_groundt.permute(1, 0, 2)
            
            output_batch = self.model(batch_boxes, batch_images, batch_motions, batch_odometery)
            loss = self.loss_fn(output_batch, batch_groundt)
            mae, _ = self.metrics(output_batch, batch_groundt)
            
            summary_batch ={
                            'loss' : loss.item(),
                            'mae' : mae.item()}
            val_summ.append(summary_batch)
        
        metrics_mean = {metric:np.mean([x[metric] for x in val_summ]) for metric in val_summ[0]} 
        return metrics_mean
    
    def train_and_eval(self, restore_from=None):
        """Train the model and evaluate every epoch.
        Args:
            restore_from: (str) directory or file containing weights to restore the graph
        """
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(self.log_dir, "exp_{}".format(self.params.exp_num) , current_time, 'train_summaries')
        eval_log_dir = os.path.join(self.log_dir, "exp_{}".format(self.params.exp_num) , current_time, 'eval_summaries')
        save_dir = os.path.join(self.log_dir, "exp_{}".format(self.params.exp_num) , current_time)
        
        self.train_summary_writer = SummaryWriter(log_dir=train_log_dir)
        self.eval_summary_writer = SummaryWriter(log_dir=eval_log_dir)
        
        begin_at_epoch = 0
        best_val_acc = 1
        train_size = len(self.train_data['images'])
        eval_size = len(self.val_data['images'])
        # TRAINING MAIN LOOP
        # ----------------------------------------------------------------------
        print("[INFO] training started ...")
        # loop over the number of epochs
        epochStart = time.time()
        for epoch in range(begin_at_epoch, begin_at_epoch + self.params.num_steps):
            # Compute number of batches in one epoch (one full pass over the training set)
            num_train_steps = int(np.ceil(train_size / self.params.batch_size))
            num_eval_steps = int(np.ceil(eval_size / self.params.batch_size))
            print("[INFO] Epoch {0:d}/{0:d}".format(epoch + 1, self.params.num_steps))
            # TRAIN SESSION ----------------------------------------------------------------------
            train_data_iterator = data_iterator(self.train_data, self.params , shuffle=True)   
            self.train_step(train_data_iterator, num_train_steps)
    
            self.train_summary_writer.add_scalar('Loss/train', self.metrics_mean['loss'], epoch+1)
            self.train_summary_writer.add_scalar('Accuracy/train', self.metrics_mean['mae'], epoch+1)
            
            # EVALUATION SESSION ----------------------------------------------------------------------
            eval_data_iterator = data_iterator(self.val_data, self.params , shuffle=False) 
            valid_metric = self.valid_step(eval_data_iterator, num_eval_steps)
            
            self.eval_summary_writer.add_scalar('Loss/valid', valid_metric['loss'], epoch+1)
            self.eval_summary_writer.add_scalar('Accuracy/valid', valid_metric['mae'], epoch+1)
            
            val_acc = valid_metric['loss']
            is_best = val_acc <= best_val_acc
            
            # Save weights
            save_checkpoint({'epoch': epoch + 1,
                               'state_dict': self.model.state_dict(),
                               'optim_dict': self.optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=save_dir)
            
            # If best_eval, best_save_path
            if is_best:
                print("[INFO] Found new best accuracy")
                best_val_acc = val_acc
                best_json_path = os.path.join(save_dir, "best_acc_model")
                torch.save(self.model.state_dict(), best_json_path)
                print("[INFO] best model saved")
                # save_dict_to_json(val_metrics, best_json_path)
        final_path = os.path.join(save_dir, "final_model")
        torch.save(self.model.state_dict(), final_path)
        print("[INFO] final model saved at {}".format(final_path))