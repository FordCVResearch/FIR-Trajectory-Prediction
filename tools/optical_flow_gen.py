import numpy as np
import cv2 as cv
import pandas as pd
from collections import Counter
from collections import namedtuple

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

SEQ_PATH = './data/data_by_seq/seq_29.csv'
SAVE_PATH = './data/of_by_seq/seq_29/'
IMG_PATH = './Dataset/Images/'

seq_data = pd.read_csv(SEQ_PATH)
grouped_test_data = split(seq_data, 'filename')

first =  grouped_test_data[0].filename
img_first = cv.imread(IMG_PATH+first+'.jpg')
hsv = np.zeros_like(img_first)
hsv[...,0] = 0
hsv[...,1] = 0
hsv[...,2] = 0
bgr_flow = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imwrite(SAVE_PATH+first+'.jpg', bgr_flow)

flow_list =[]   
for idx in range(len(grouped_test_data)-1):
    first_img =  grouped_test_data[idx].filename
    second_img =  grouped_test_data[idx+1].filename
    
    img_1 = cv.imread(IMG_PATH+first_img+'.jpg')
    hsv = np.zeros_like(img_1)

    if img_1.shape[2] == 3:
        img_1 = cv.cvtColor(img_1,cv.COLOR_BGR2GRAY)
    
    img_2 = cv.imread(IMG_PATH+second_img+'.jpg')
    if img_2.shape[2] == 3:
        img_2 = cv.cvtColor(img_2,cv.COLOR_BGR2GRAY)
        
    hsv[...,1] = 255
        
    flow = cv.calcOpticalFlowFarneback(img_1, img_2, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr_flow = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imwrite(SAVE_PATH+grouped_test_data[idx+1].filename+'.jpg', bgr_flow)

print(' Successfully created the vide file')

