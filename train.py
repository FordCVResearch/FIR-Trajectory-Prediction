"""
Script to train the encoder-decoder model using the training dataset
"""
import argparse
import os
from packaging import version

import torch
import torch.optim as optim

from model.utils import Params, define_config
from data.data_loader import DataLoader
from model.network.base_model import Encoder_Decoder

from model.metric import accuracy, loss
from model.train.training import Train_and_Evaluate

# Set the cuda device as environmental variable. Usage of set_device() is discouraged in favor of device!! 
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='./experiments/run_2',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./dataset',
                    help="Directory containing the dataset")

parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

parser.add_argument('--log_dir', default="./log",
                    help="log directory for the trained model")

parser.add_argument('--mode', default='train', 
                    help="train or test mode")

if __name__ == '__main__':
    
    args = parser.parse_args()
    
    # check pytorch version be greater than 1.0
    print("Pytorch version: ", torch.__version__)
    assert version.parse(torch.__version__).release[0] >= 1, \
    "This notebook requires pytorch 1.0 or above."
    
    # Load the parameters from json file
    json_path = os.path.join(args.experiment_dir, 'net_params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(device=0)
    print("Using cuda: {}, device: {}".format(params.cuda, device_name))

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        
    # check if the log file is available
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # check if the data is available
    assert os.path.exists(args.data_dir), "No data file found at {}".format(args.data_dir)
    print('[INFO] data preprocessing started. This could take a while!')
    data_loader = DataLoader(args.data_dir)
    data = data_loader.load_data()
    train_data = data[0]
    val_data = data[1]
    data_by_seq_train = data_loader.split_by_seq(train_data)
    data_by_seq_valid = data_loader.split_by_seq(val_data)
    print('[INFO] data preprocessing done! {0} training sequences and {1} validation sequences'.format(len(data_by_seq_train['images']), 
                                                                                                       len(data_by_seq_valid['images'])))
    print('=================================================')
    
    model_config = define_config(params)
    if params.cuda:
        model = Encoder_Decoder(model_config['encoder_config'], 
                                model_config['decoder_config'], 
                                prediction_length=params.prediction_length, 
                                roi_size=(params.roi_size, params.roi_size), 
                                image_size=(params.con_image_size, params.con_image_size)).cuda()
    else:
        model = Encoder_Decoder(model_config['encoder_config'], 
                        model_config['decoder_config'], 
                        prediction_length=params.prediction_length, 
                        roi_size=(params.roi_size, params.roi_size), 
                        image_size=(params.con_image_size, params.con_image_size))
        
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = loss.loss_fn
    metrics = accuracy.accuracy_fn
    print('[INFO] model created...')
    print('=================================================')
    train_job = Train_and_Evaluate(model, 
                                   loss_fn, 
                                   optimizer, 
                                   metrics, 
                                   params, 
                                   data_by_seq_train,
                                   data_by_seq_valid,
                                   args.log_dir)
    train_job.train_and_eval()
    print('[INFO] Done!')