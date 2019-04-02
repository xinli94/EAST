import collections
import cv2
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

root_path = '/data5/thomas/data-pipeline/MSRA-TD500/'
txt_folder = '/data5/xin/MSRA-TD500/text/'
# image, x1, y1, x2, y2, x3, y3, x4, y4, label
csv_path = '/data5/xin/MSRA-TD500/msra_td500.csv'
# image_path, txt_path
east_csv_path = '/data5/xin/MSRA-TD500/all.csv'

if not os.path.exists(txt_folder):
    os.makedirs(txt_folder)

count = total = 0
records = []
files = collections.defaultdict(list)

for prefix in ['train', 'test']:
    for image_path in tqdm(glob.glob(os.path.join(root_path, prefix, '*.JPG'))):
        total += 1
        gt_path = os.path.splitext(image_path)[0] + '.gt'
        if not os.path.exists(gt_path):
            count += 1
            continue

        with open(gt_path) as f:
            for line in f.readlines():
                # index, difficult, x, y, w, h, theta
                _, _, x, y, w, h, theta = [float(item) for item in line.strip().split() if item]
                x_c, y_c = x + w/2.0, y + h/2.0
                theta = theta / np.pi * 180

                box = cv2.boxPoints(((x_c, y_c), (w, h), theta))
                polygon = np.reshape(box, [-1, ]).tolist()
                records.append([image_path] + polygon + ['text'])
                files[image_path].append(polygon + ['text'])

print('Skipped {}/{}'.format(count, total))

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
