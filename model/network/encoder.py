""" Defines the multi-stream encoder architecture """

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import motion_encoder, contextual_encoder, trajectory_encoder

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    
    """An implementation of the multimodal encoder.
        Args:
        traj_config (dict): input config for trajectory encoder
        contex_config (dict): input config for contextual encoder
        motion_config (dict): input config for motion encoder
        use_contex (bool): wether to use contextual encoder
        use_motion (bool): wether to use motion encoder
        use_mlp (bool): wether to use mlp after concatination
        mlp_kernel_size (int): kernel size for the mlp layer
    """
       
    def __init__(self, 
                 traj_config,
                 contex_config,
                 motion_config,
                 use_contex=True,
                 use_motion=True,
                 mlp_kernel_size=256,
                 use_mlp=True):   
        super(Encoder, self).__init__()  
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        
        self.traj_config = traj_config
        self.contex_config = contex_config
        self.motion_config = motion_config
        self.use_contex = use_contex
        self.use_motion = use_motion
        self.mlp_kernel_size = mlp_kernel_size
        self.use_mlp = use_mlp
        
        # setup the trajectory encoder
        self.traj_enc= trajectory_encoder.TrajectoryEncoder(self.traj_config['input_size'], 
                                                             self.traj_config['hidden_size'], 
                                                             self.traj_config['num_layers'],
                                                             embeded_encoding=self.traj_config['embeded_encoding'],
                                                             out_kernel_size=self.traj_config['out_kernel_size'])
        self.con_enc= contextual_encoder.ContextEncoder(self.contex_config['image_size'], 
                                                             self.contex_config['num_filters'], 
                                                             self.contex_config['convLSTM_kernel_size'],
                                                             self.contex_config['output_size'],
                                                             cnn_model=self.contex_config['cnn_model'],
                                                             num_layers= self.contex_config['num_layers'],
                                                             use_MLP=self.contex_config['use_MLP'])
        self.mot_enc= motion_encoder.MotionEncoder(self.motion_config['map_size'], 
                                                    self.motion_config['num_filters'],
                                                    self.motion_config['convLSTM_kernel_size'],
                                                    self.motion_config['output_size'], 
                                                    num_layers= self.motion_config['num_layers'],
                                                    use_MLP=self.motion_config['use_MLP'])
        
    def forward(self, x_in_traj, x_in_cont, x_in_mot):
        """final multi-stream encoder definition.
        Args:
            x_in_traj (FloatTensor): `[T (query), #B , bb_size=4]`
            x_in_cont (FloatTensor): `[T (query), #B , image_size=(C, H, W)]`
            x_in_mot (FloatTensor): `[T (query), #B , map_size=(2, H, W)]`
        Returns:
            x_out (FloatTensor): `[#B, output_size]`
        """
        
        traj_h, traj_c = self.traj_enc.forward(x_in_traj)
        self.combined_h =  traj_h
        self.combined_c =  traj_c
        if self.use_contex:
            cont_h, cont_c = self.con_enc.forward(x_in_cont)
            self.combined_h = torch.cat((self.combined_h, cont_h), dim=1)
            self.combined_c = torch.cat((self.combined_c, cont_c), dim=1)
            
        if self.use_motion:    
            mot_h, mot_c = self.mot_enc.forward(x_in_mot)
            self.combined_h = torch.cat((self.combined_h, mot_h), dim=1)
            self.combined_c = torch.cat((self.combined_c, mot_c), dim=1)
            
        if self.use_mlp:
            mlp = torch.nn.Linear(self.combined_h.shape[1] ,self.mlp_kernel_size)
            activation = torch.nn.ReLU()
            
            mlp = mlp.cuda() #cuda bug fix needed!!!
            self.h_out = mlp(self.combined_h)
            self.h_out = activation(self.h_out)
            
            self.c_out = mlp(self.combined_c)
            self.c_out = activation(self.c_out)
        else:
            self.h_out = self.combined_h
            self.c_out = self.combined_c
        
        return (self.h_out, self.c_out)