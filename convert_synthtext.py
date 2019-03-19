import numpy as np
import os
import pandas as pd
import scipy.io as sio

root_path = '/data5/thomas/data-pipeline/SynthText/'

gt_path = os.path.join(root_path, 'gt.mat')
data = sio.loadmat(gt_path)

records = []
n = len(data['wordBB'][0])
for image_name, raw_text, raw_bboxes in zip(data['imnames'][0], data['txt'][0], data['wordBB'][0]):
    image_path = os.path.join(root_path, image_name[0])
    raw_bboxes = raw_bboxes.reshape([2,4,-1])
    
    text = list(filter(lambda x: x.strip(), ' '.join(raw_text).split()))
    assert len(text) == raw_bboxes.shape[2]
    
    for bidx in range(raw_bboxes.shape[2]):
        bboxes = raw_bboxes[:,:,bidx]
        records.append([image_path] + list(bboxes.T.flatten()) + [text[bidx]])