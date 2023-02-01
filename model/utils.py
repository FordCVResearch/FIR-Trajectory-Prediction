"""Unility functions for general purposes."""

import json
import  os
import logging
import torch
import shutil

from pathlib import Path
from collections import namedtuple
import numpy as np


def mkdir_join(path, *dir_name, rank=0):
    """Concatenate root path and 1 or more paths, and make a new directory if the directory does not exist.
    Args:
        path (str): path to a directory
        rank (int): rank of current process group
        dir_name (str): a directory name
    Returns:
        path to the new directory
    """
    p = Path(path)
    if not p.is_dir() and rank == 0:
        p.mkdir()
    for i in range(len(dir_name)):
        # dir
        if i < len(dir_name) - 1:
            p = p.joinpath(dir_name[i])
            if not p.is_dir() and rank == 0:
                p.mkdir()
        elif '.' not in dir_name[i]:
            p = p.joinpath(dir_name[i])
            if not p.is_dir() and rank == 0:
                p.mkdir()
        # file
        else:
            p = p.joinpath(dir_name[i])
    return str(p.absolute())

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. 
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def split(df, group):
    """
    split the Pandas dataframe into two groups including filenam and objects
    Inputs:
    - df: Pandas dataframe.e file
    - group: a string with the name of the first group name 
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def process_odometery(seq_odom_path):
    with open(seq_odom_path) as filestream: 
        for _, line in enumerate(filestream):
            currentline = line.split(" ")
            dx= float(currentline[1])
            dy= float(currentline[2])
            # yaw (z-axis rotation)
            # double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
            # double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
            # angles.yaw = std::atan2(siny_cosp, cosy_cosp);
            theta = np.arctan2(2*(float(currentline[4])*float(currentline[7][:-2]) + float(currentline[5])*float(currentline[6])),
                               1- 2*(float(currentline[6])**2 + float(currentline[7][:-2])**2))
            return np.array(dx, dy, theta, dtype=float)

def find_center_normal(row, width, height):
    xmin = row['x1'] / width
    xmax = row['x2'] / width
    ymin = row['y2'] / height
    ymax = row['y1'] / height
    
    xc = (xmin + xmax)/2
    yc = (ymin + ymax)/2
    w = np.absolute(xmax - xmin)
    h = np.absolute(ymax - ymin)
    
    return np.array([xc,yc,w,h])

def define_config(params):
    
    model_config= {}
    traj_config = {'input_size': params.traj_input_size,
                'hidden_size': params.traj_hidden_size,
                'num_layers': params.traj_num_layer,
                'embeded_encoding': params.traj_emb_encod,
                'out_kernel_size': params.traj_out_size}

    motion_config = {'map_size': (params.mot_map_size, params.mot_map_size),
                'num_filters': [params.mot_filter1_size, params.mot_filter2_size],
                'convLSTM_kernel_size': (1, 1),
                'output_size': params.mot_out_size,
                'num_layers':params.mot_num_layer,
                'use_MLP': params.mot_use_MLP}
    
    contex_config = {'image_size': (3, params.con_image_size, params.con_image_size),
                'num_filters': [params.con_filter1_size, params.con_filter2_size],
                'convLSTM_kernel_size': (1, 1),
                'output_size':params.con_out_size,
                'num_layers':params.con_num_layer,
                'cnn_model': None,
                'use_MLP': params.con_use_MLP}

    encoder_config = {'traj_config': traj_config,
                    'contex_config': contex_config,
                    'motion_config' :motion_config,
                    'use_mlp': True,
                    'mlp_kernel_size': params.encoder_mlp}

    decoder_config = {'input_dim': params.decoder_input_dim, #must be 2x the mlp_size
                    'hidden_dim':  params.decoder_hidden_dim,
                    'mlp_size': params.decoder_mlp_size}
    
    model_config['encoder_config'] = encoder_config
    model_config['decoder_config'] = decoder_config
    
    return model_config

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)
    
def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
