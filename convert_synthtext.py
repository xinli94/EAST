import collections
import numpy as np
import os
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

root_path = '/data5/thomas/data-pipeline/SynthText/'
txt_folder = '/data5/xin/SynthText/text/'
# image, x1, y1, x2, y2, x3, y3, x4, y4, label
csv_path = 'synthtext.csv'
# image_path, txt_path
east_csv_path = '/data5/xin/SynthText/all.csv'

if not os.path.exists(txt_folder):
    os.makedirs(txt_folder)

gt_path = os.path.join(root_path, 'gt.mat')
data = sio.loadmat(gt_path)

records = []
files = collections.defaultdict(list)
total = len(data['imnames'][0])
innames, txts, wordBBs = data['imnames'][0], data['txt'][0], data['wordBB'][0]

for image_name, raw_text, raw_rboxes in tqdm(zip(innames, txts, wordBBs), total=total):
    image_path = os.path.join(root_path, image_name[0])
    raw_rboxes = raw_rboxes.reshape([2,4,-1])
    
    text = list(filter(lambda x: x.strip(), ' '.join(raw_text).split()))
    assert len(text) == raw_rboxes.shape[2]
    
    for bidx in range(raw_rboxes.shape[2]):
        rboxes = raw_rboxes[:,:,bidx].T.flatten()
        records.append([image_path] + list(rboxes) + [text[bidx]])
        files[image_path].append(list(rboxes) + [text[bidx]])

pd.DataFrame.from_records(records).to_csv(csv_path, header=None, index=None)

##################### icdar format ########################
records_out = []
for image_path, data in tqdm(files.items()):
    txt_path = os.path.join(txt_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    # save to txt
    pd.DataFrame.from_records(data).to_csv(txt_path, header=None, index=None)
    records_out.append([image_path, txt_path])

# East training input csv
pd.DataFrame.from_records(records_out).to_csv(east_csv_path, header=None, index=None)
