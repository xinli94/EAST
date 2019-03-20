import collections
import json
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

root_path = '/data5/thomas/data-pipeline/coco_raw/'
txt_folder = '/data5/xin/cocoText/text/'
# image, x1, y1, x2, y2, x3, y3, x4, y4, label
csv_path = '/data5/xin/cocoText/cocotext.csv'
# image_path, txt_path
east_csv_path = '/data5/xin/cocoText/all.csv'

if not os.path.exists(txt_folder):
    os.makedirs(txt_folder)

def _image_path(image_name):
    for prefix in ['train2014', 'val2014']:
        image_path = os.path.join(root_path, prefix, image_name)
        if os.path.exists(image_path):
            return image_path
    return None

gt_path = os.path.join(root_path, 'coco-text/COCO_Text.json')
with open(gt_path) as f:
    data = json.load(f)

records = []
files, sets = collections.defaultdict(list), {}
images, anns = data['imgs'], data['anns']

count = 0
for info in tqdm(anns.values()):    
    image_id = str(info['image_id'])
    image_name, which_set = images[image_id]['file_name'], images[image_id]['set']
    image_path = _image_path(image_name)
    
    if image_path is None or 'utf8_string' not in info:
        count += 1
        # print('Skip {}: {}'.format(count, image_path))
        continue

    polygon, label = info['polygon'], info['utf8_string']
    records.append([image_path] + polygon + [label])
    files[image_path].append(list(polygon) + [label])
    sets[image_path] = which_set
print('Skipped {}/{}'.format(count, len(anns.keys())))

pd.DataFrame.from_records(records).to_csv(csv_path, header=None, index=None)

##################### icdar format ########################
records_out = []
for image_path, data in tqdm(files.items()):
    txt_path = os.path.join(txt_folder, os.path.splitext(os.path.basename(image_path))[0] + '_nohash_' + sets[image_path] + '.txt')
    # save to txt
    pd.DataFrame.from_records(data).to_csv(txt_path, header=None, index=None)
    records_out.append([image_path, txt_path])

# East training input csv
pd.DataFrame.from_records(records_out).to_csv(east_csv_path, header=None, index=None)
