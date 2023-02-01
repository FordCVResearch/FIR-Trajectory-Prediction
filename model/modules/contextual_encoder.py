""" Defines the encoder architecture for the the contextual image input """

import logging
import torch
import torch.nn as nn
import torchvision.models as models

from model.modules.utility_layers import  Identity, ConvLSTM, ConvGRU

logger = logging.getLogger(__name__)

class ContextEncoder(nn.Module):
    
    """A single layer of the contextual encoder.
        Args:
        image_size (tuple): size of the input image
        num_filters (int): number of filters in ConvLSTM
        kernel_size (tuple): size of the kernel in ConvLSTM 
        output_size (int): size of the output FC layer
        cnn_model (str): backbone model name
        num_layers (int): number of recurrent stacks (either GRU or LSTM)
        use_gru (bool): whether to use GRU or LSTM
        use_dopout_lstm (bool): whether to use Dropout layer on the outputs of 
            each LSTM layer except the last layer
        use_LeakyRelu (bool): whether to use LeakyReLU or not
    """
       
    def __init__(self, 
                 image_size,
                 num_filters, 
                 convLSTM_kernel_size,
                 output_size,
                 cnn_model=None,
                 num_layers=1, 
                 padding=0,
                 stride=1,
                 use_gru=False,
                 cnn_pretrained = True,
                 use_dopout_lstm= False,
                 use_LeakyRelu= False,
                 use_MLP=True):   
        super(ContextEncoder, self).__init__()
        logger.info('===== Initialize %s =====' % self.__class__.__name__)              
       
        self.image_size = image_size
        self.num_filters = num_filters
        self.kernel_size = convLSTM_kernel_size 
        self.output_size = output_size
        self.cnn_model = cnn_model
        self.num_layers = num_layers
        self.cnn_pretrained = cnn_pretrained
        self.use_gru = use_gru
        self.use_dopout_lstm = use_dopout_lstm
        self.use_LeakyRelu = use_LeakyRelu
        self.use_MLP = use_MLP
        
        self.cnn_out_size = {'vgg': (512,7,7),
                             'resnet': (2048, 1 ,1),
                             'inception': (2048, 1, 1)}
        
        # CNN encoder layer
        if self.cnn_model is not None:
            if cnn_model == "vgg":
                self.cnn_in = models.vgg16(pretrained=self.cnn_pretrained)
                # throw the classification layers away
                self.cnn_in.classifier = Identity() # (7*7*512)
            elif cnn_model == "resnet":
                self.cnn_in = models.resnet50(pretrained=self.cnn_pretrained)
                # throw the classification layers away 
                self.cnn_in.fc = Identity()  # (1*1*2048)
            elif cnn_model == "inception":
                self.cnn_in = models.inception_v3(pretrained=self.cnn_pretrained)
                # throw the classification layers away
                self.cnn_in.fc = Identity() # (1*1*2048)
            else:
                raise ValueError('`Model name` is not supported')
            self.back_out_size = self.cnn_out_size[self.cnn_model]
        else:
            self.cnn_in = Identity()
            self.back_out_size = list(self.image_size) # image will directly feed into ConvLSTM 
            
        # ConvGRU stack
        if self.use_gru:
            self.enc_rnn = ConvGRU(self.back_out_size[0], self.num_filters,num_layers =self.num_layers)
        # ConvLSTM stack
        else:
            self.enc_rnn = ConvLSTM(self.back_out_size[0], self.num_filters, self.kernel_size, 
                                    num_layers =self.num_layers, stride=stride, padding=padding)     
        # output embedding layer   
        if self.use_MLP:
            fc_in_size = self.back_out_size[1] * self.back_out_size[2] * self.num_filters[-1]
            self.fc_out = torch.nn.Linear(fc_in_size ,self.output_size)
        # activation
        if use_LeakyRelu:
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            self.activation = torch.nn.ReLU()
        
        
    def forward(self, x_in):
        
        """contextual encoder layer definition.
        Args:
            x_in (FloatTensor): `[T (query), #B , image_size=(C, H, W)]`
        Returns:
            x_out (FloatTensor): `[#B, output_size]`
            
        [T,#B,image_size]
                | ------------------|
                V                   |
            Backbone CNN            |
        [T,#B,feature_map_size]     |
                |                   |
                |<------------------|
                V
            ConvLSTM Stack
        [#B,convLSTM_kernel_size,(feature_map_w, feature_map_h)] ##same padding is used to preserve the size 
                |
                V
              Flatten
        [#B, convLSTM_kernel_size * feature_map_w * feature_map_h]   
                |
                |------------------> Concatination
                V
                FC
        [#B, output_size]
        """
        # residual = x_in  
        T = x_in.shape[0]
        B = x_in.shape[1]
        # set the output size after the backbone cnn net
        x_in_sequence = torch.empty(T, B, self.back_out_size[0], self.back_out_size[1], self.back_out_size[2], dtype=torch.float) #!!!!!! fix device att
        x_in_sequence = x_in_sequence.cuda()
        # cnn encoder layer
        for seq_idx in range (T):
            x_in_seq = self.cnn_in(x_in[seq_idx])
            x_in_sequence[0,:] = x_in_seq.view(-1, self.back_out_size[0], self.back_out_size[1], self.back_out_size[2])
            
        # ConvLSTM layer
        _, self.last_states = self.enc_rnn.forward(x_in_sequence)
        self.last_h = self.last_states[0]
        self.last_c = self.last_states[1]
        
        # Flatten the lstm output
        self.h_flat = torch.flatten(self.last_h, start_dim=1)
        self.c_flat = torch.flatten(self.last_c, start_dim=1)
        
        assert self.h_flat.shape[-1] == self.c_flat.shape[-1], "last layer hidden sizes missmatch the cell state sizes"
        
        if self.use_MLP:
            # transform to encoded vector
            self.h_out = self.fc_out(self.h_flat)
            self.h_out = self.activation(self.h_out)
            
            self.c_out = self.fc_out(self.c_flat)
            self.c_out = self.activation(self.c_out)
        else:
            self.h_out = self.h_flat
            self.c_out = self.c_flat

        return (self.h_out, self.c_out)

        
    
        
        
        
    