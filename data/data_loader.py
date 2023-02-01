import os
import random
import torch
import cv2
import pandas as pd
import numpy as np

from torch.autograd import Variable
from model.utils import Params, split, process_odometery, find_center_normal

class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params
    """
    def __init__(self, dataset_dir, params=None):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have the entire dataset on data_dir before using this
        class.
        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params to params.
        """
        json_path = os.path.join(dataset_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
        self.dataset_params = Params(json_path) 
        
        self.images_path = os.path.join(dataset_dir, 'images')
        self.seqs_path = os.path.join(dataset_dir, 'seq')
        
    def process_image(self, image_filename):
        image_np = cv2.imread(image_filename)
        # resize the image to the desired input size for the contextual encoder
        image_resize = cv2.resize(image_np, (self.dataset_params.image_width, self.dataset_params.image_height))
        # permutat the dimention order to be consistent with torch (C, W, H)
        return np.transpose(image_resize, (2,1,0))

    
    def process_motion(self, motion_filename):
        motion_np = cv2.imread(motion_filename)
        motion_np = cv2.resize(motion_np, (640,480))
        motion_np_2d = motion_np[:,:,[0,2]]
        # permutat the dimention order to be consistent with torch (C, W, H)
        return np.transpose(motion_np_2d, (2,1,0))
        
    def data_process(self, seq_data_path, seq_motion_path, seq_odom_path=None):
        data_pd = pd.read_csv(seq_data_path)
        grouped_data = split(data_pd, 'filename')
        img_seq = []
        mot_seq = []
        box_seq = []
        odt_seq = []
        for Idx, image in  enumerate(grouped_data):
            image_filename = os.path.join(self.images_path, image.filename+'.jpg')
            image_np = self.process_image(image_filename)
            motion_filename = os.path.join(seq_motion_path,image.filename+'.jpg')
            motion_np = self.process_motion(motion_filename)
            width = image.object.width.values[0]
            height = image.object.height.values[0]
            if seq_odom_path is not None:
                odometery = process_odometery(seq_odom_path)
            else:   
                odometery = np.zeros(3)
            for _, row in image.object.iterrows():
                box = find_center_normal(row, width, height)
                img_seq.append(image_np) 
                mot_seq.append(motion_np)
                box_seq.append(box)
                odt_seq.append(odometery)
            if Idx%100 == 0:
                print('%d features are processed'% Idx)
        return img_seq, mot_seq, box_seq, odt_seq
                
    def load_data(self):
        """
        Loads the data for each type in types from data_dir.
        Args:
        Returns:
            data: (dict) contains the data with labels for each type in types
        """
        tarin_data ={}
        tarin_data['images'] = []
        tarin_data['motions'] = []
        tarin_data['boxes'] = []
        tarin_data['odometery'] = []
        tarin_data['ground_truth'] = []

        valid_data ={}
        valid_data['images'] = []
        valid_data['motions'] = []
        valid_data['boxes'] = []
        valid_data['odometery'] = []
        valid_data['ground_truth'] = []
        
        test_data ={}
        test_data['images'] = []
        test_data['motions'] = []
        test_data['boxes'] = []
        test_data['odometery'] = []
        test_data['ground_truth'] = []
        
        for seq_idx in range(self.dataset_params.seq_num):
            seq_path = os.path.join(self.seqs_path, 'seq_{:02d}'.format(seq_idx))
            seq_data_path = os.path.join(seq_path, 'seq_{:02d}.csv'.format(seq_idx))
            seq_motion_path = os.path.join(seq_path, 'motions')
            seq_odom_path = os.path.join(seq_path, 'KeyFrameTrajectory.txt')
            
            if seq_idx == self.dataset_params.eval_seq:
                eval_dt = self.data_process(seq_data_path, seq_motion_path)
                valid_data['images'].append(eval_dt[0])
                valid_data['motions'].append(eval_dt[1])
                valid_data['boxes'].append(eval_dt[2])
                valid_data['odometery'].append(eval_dt[3])
                
            elif seq_idx == self.dataset_params.test_seq:
                test_dt = self.data_process(seq_data_path, seq_motion_path)
                test_data['images'].append(test_dt[0])
                test_data['motions'].append(test_dt[1])
                test_data['boxes'].append(test_dt[2])
                test_data['odometery'].append(test_dt[3])
                
            else:
                train_dt = self.data_process(seq_data_path, seq_motion_path)
                tarin_data['images'].append(train_dt[0])
                tarin_data['motions'].append(train_dt[1])
                tarin_data['boxes'].append(train_dt[2])
                tarin_data['odometery'].append(train_dt[3])
                
        if not (len(tarin_data['images']) == len(tarin_data['motions']) == len(tarin_data['boxes']) == len(tarin_data['odometery'])):
            raise ValueError('feature dimentions does not match')
                
        return tarin_data, valid_data, test_data
    
    def split_by_seq(self, data):
        seq_num = len(data['images']) 
        l = self.dataset_params.encoding_len
        split_data = {}
        img_seq = []
        mot_seq = []
        box_seq = []
        odt_seq = []
        grt_seq = []
        for seq_idx in range(seq_num):
            seq_data_num = int(np.floor( len(data['images'][seq_idx])/ self.dataset_params.encoding_len))
            for i in range (seq_data_num-1):
                img_seq += [np.array(data['images'][seq_idx][i*l:(i+1)*l])]
                mot_seq += [np.array(data['motions'][seq_idx][i*l:(i+1)*l])]
                box_seq += [np.array(data['boxes'][seq_idx][i*l:(i+1)*l])]
                odt_seq += [np.array(data['odometery'][seq_idx][i*l:(i+1)*l])]
                grt_seq += [np.array(data['boxes'][seq_idx][(i+1)*l:(i+2)*l])]
        
        split_data['images'] = img_seq
        split_data['motions'] = mot_seq
        split_data['boxes'] = box_seq
        split_data['odometery'] = odt_seq
        split_data['groundt'] = grt_seq
        
        return split_data
    
def data_iterator(data, params, shuffle=False):
    """
    Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
    pass over the data.
    Args:
        data: (dict) contains data which has keys 'data'
        params: (Params) hyperparameters of the training process.
        shuffle: (bool) whether the data should be shuffled
    """
    data_size = len(data['images'])
    
    order = list(range(data_size))
    if shuffle:
        random.seed(230)
        random.shuffle(order)
    
    for i in range((data_size+1)//params.batch_size):
        batch_images = [data['images'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
        batch_motions = [data['motions'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
        batch_boxes = [data['boxes'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
        batch_odometery = [data['odometery'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
        batch_groundt = [data['groundt'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
        
        batch_max_len = max([len(s) for s in batch_images])
        
        batch_images_np = np.zeros((len(batch_images), batch_max_len, 3, params.con_image_size, params.con_image_size))
        batch_motions_np = np.zeros((len(batch_motions), batch_max_len, 2, 640, 480))
        batch_boxes_np = np.zeros((len(batch_boxes), batch_max_len, params.traj_input_size))
        batch_odometery_np = np.zeros((len(batch_odometery), batch_max_len, 3))
        batch_groundt_np = np.zeros((len(batch_groundt), batch_max_len, params.traj_input_size))
        
        # copy the data to the numpy array
        for j in range(len(batch_images)):
            batch_images_np[j][:] = batch_images[j]
            batch_motions_np[j][:] = batch_motions[j]
            batch_boxes_np[j][:] = batch_boxes[j]
            batch_odometery_np[j][:] = batch_odometery[j]
            batch_groundt_np[j][:] = batch_groundt[j]
            
        # since all data are indices, we convert them to torch LongTensors
        batch_images_t =  torch.FloatTensor(batch_images_np)
        batch_motions_t =  torch.FloatTensor(batch_motions_np)
        batch_boxes_t =  torch.FloatTensor(batch_boxes_np)
        batch_odometery_t =  torch.FloatTensor(batch_odometery_np)
        batch_groundt_t =  torch.FloatTensor(batch_groundt_np)
        
        # shift tensors to GPU if available
        if params.cuda:
            batch_images_t = batch_images_t.cuda()
            batch_motions_t = batch_motions_t.cuda()
            batch_boxes_t = batch_boxes_t.cuda()
            batch_odometery_t = batch_odometery_t.cuda()
            batch_groundt_t = batch_groundt_t.cuda()
            
        # convert them to Variables to record operations in the computational graph
        batch_images_t= Variable(batch_images_t)
        batch_motions_t= Variable(batch_motions_t)
        batch_boxes_t= Variable(batch_boxes_t)
        batch_odometery_t= Variable(batch_odometery_t)
        batch_groundt_t= Variable(batch_groundt_t)
         
        yield batch_images_t, batch_motions_t, batch_boxes_t, batch_odometery_t, batch_groundt_t