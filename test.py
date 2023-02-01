"""
Script to test the encoder-decoder model using the test dataset
"""
import argparse
import os
from packaging import version

import torch

from model.utils import Params, define_config
from data.data_loader import DataLoader
from model.network.base_model import Encoder_Decoder

from model.metric import accuracy, loss
from model.train import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', default='./experiments/run_1',
                    help="Experiment directory containing params.json")

parser.add_argument('--data_dir', default='./dataset',
                    help="Directory containing the dataset")

parser.add_argument('--model_dir', default="./log/exp_1/20210811-124121/",
                    help="saved model directory for the trained model")

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
    print("Using cuda: {}".format(params.cuda))
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
        
    # check if the data is available
    assert os.path.exists(args.data_dir), "No data file found at {}".format(args.data_dir)
    print('[INFO] data preprocessing started. This could take a while!')
    data_loader = DataLoader(args.data_dir)
    data = data_loader.load_data()
    test_data = data[2]
    data_by_seq_test = data_loader.split_by_seq(test_data)
    print('[INFO] data preprocessing done! {0} test sequences'.format(len(data_by_seq_test['images'])))
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
    model_path = os.path.join(args.model_dir,'best_acc_model')
    model.load_state_dict(torch.load(model_path))
    print('[INFO] model loaded...')
    print('=================================================')
    loss_fn = loss.loss_fn
    metrics = accuracy.accuracy_fn
    
    test_metrics = evaluate.test(model, 
                loss_fn, 
                metrics, 
                params, 
                data_by_seq_test,
                args.model_dir,
                train_size=len(data_by_seq_test['images']))
    # save_path = os.path.join(args.model_dir, "metrics_test.json")
    # utils.save_dict_to_json(test_metrics, save_path)
    print('[INFO] Done!')