""" Defines the entire architecture for multi-stream encoder decoder network """

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_pool

from model.network.encoder import  Encoder
from model.network.decoder import Decoder

logger = logging.getLogger(__name__)

class Encoder_Decoder(nn.Module):

    """An implementation of the encoder decoder network.
        Args:
        encoder_config (dict): input config for the encoder
        decoder_config (dict): input config for the decoder
        prediction_length (int): length of the future sequence for prediction
        roi_size (tuple): ROI pooling output size 
        image_size (tuple): input size for the image (W, H)
    """
    def __init__(self, 
                 encoder_config,
                 decoder_config,
                 prediction_length,
                 roi_size,
                 image_size):   
        super(Encoder_Decoder, self).__init__()  
        logger.info('===== Initialize %s =====' % self.__class__.__name__)
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.prediction_length = prediction_length
        self.roi_size = roi_size
        self.image_size = image_size
    
        self.encoder_model = Encoder(self.encoder_config['traj_config'],
                                self.encoder_config['contex_config'],
                                self.encoder_config['motion_config'], 
                                use_contex=True,
                                use_motion= True,
                                use_mlp=encoder_config['use_mlp'],
                                mlp_kernel_size=encoder_config['mlp_kernel_size'])
        self.decoder_model = Decoder(self.decoder_config,
                                     prediction_len= self.prediction_length)
        
    def forward(self, boxes, images, optical_flows, odometry):
        """final model definition.
        Args:
            boxes (FloatTensor): `[T (query), #B , bb_size=4(first column is the box index)]`
            images (FloatTensor): `[T (query), #B , image_size=(C, H, W)]`
            optical_flows (FloatTensor): `[T (query), #B , map_size=(2, H, W)]`
            odometry (FloatTensor): `[T(query), #B, (Vx, Vy, Theta)]`
        Returns:
            x_out (FloatTensor): `[#B, t, bb_size=4]`
        """
        # some sanity checks
        self.check_multi_input_size(boxes, images, optical_flows, odometry)
        self.encoder_len = boxes.shape[0]
        
        pooled_motion_list =[]            
        # RoI pooling on the optical flows
        for t in range(self.encoder_len):
            pooled_motion = roi_pool(optical_flows[t,:,:,:], [boxes[t,:,:]], self.roi_size)
            pooled_motion_list.append(pooled_motion)
            
        roi_motion = torch.stack(pooled_motion_list, dim=0)
        
        # encoder step
        encoded_features = self.encoder_model.forward(boxes, images, roi_motion)
        # decoder step
        pred_boxes = self.decoder_model(encoded_features, odometry)
        
        return pred_boxes
               
    @staticmethod
    def check_multi_input_size(x1, x2, x3, x4):
        if not (x1.shape[0]== x2.shape[0] == x3.shape[0]):
            raise ValueError('`sequence len` must be equal for all inputs')
        if not (x1.shape[1]== x2.shape[1] == x3.shape[1]):
            raise ValueError('`batch size` must be equal for all inputs')
             