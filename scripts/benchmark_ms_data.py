import collections
import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# input_csv = '/Users/xinli/Downloads/ms_ocr_output_samples.csv'
root_path = '/data5/xin/ocr/ms_ocr_output_samples/'
input_csv = os.path.join(root_path, 'ms_ocr_output_samples.csv')
east_csv_path = os.path.join(root_path, 'all.csv')

image_folder = os.path.join(root_path, 'images')
txt_folder = os.path.join(root_path, 'text')
txt_folder8 = os.path.join(root_path, 'text8')

for folder in [image_folder, txt_folder, txt_folder8]:
    if not os.path.exists(folder):
        os.makedirs(folder)

df = pd.read_csv(input_csv)

records_bbox, records_bbox8, records_rbox = [], [], []
files = collections.defaultdict(list)

with open(os.path.join(root_path, 'download.sh'), 'w+') as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        width, height = row['width'], row['height']
        meta = json.loads(row['ms_ocr_response'])['recognitionResult']['lines']
        if not meta:
            continue

        image_name, s3_bucket = row['s3_key'], row['s3_bucket']
        s3_path = os.path.join('s3://' + s3_bucket, image_name)
        image_path = os.path.splitext(os.path.join(image_folder, image_name))[0] + '.jpg'
        f.write('aws s3 cp {} {} \n'.format(s3_path, image_path))
        
        for res in meta:
            words = res['words']
            for w in words:
                rbox, label = w['boundingBox'], w['text']
                records_rbox.append([image_path] + rbox + [label])
                files[image_path].append(rbox + [label])
                
                rbox = np.array(w['boundingBox']).reshape((4, 2)).astype(np.int32)
                min_h, max_h = rbox[:, 1].min(), rbox[:, 1].max()
                min_w, max_w = rbox[:, 0].min(), rbox[:, 0].max()
                bbox = [min_w, min_h, max_w, max_h]
                records_bbox.append([image_path, width, height] + bbox + [label])
                records_bbox8.append([image_path, min_w, min_h, max_w, min_h, max_w, max_h, min_w, max_h, label])

pd.DataFrame.from_records(records_bbox).to_csv(os.path.join(root_path, 'bbox.csv'), header=None, index=None)
pd.DataFrame.from_records(records_bbox8).to_csv(os.path.join(root_path, 'bbox8.csv'), header=None, index=None)
pd.DataFrame.from_records(records_rbox).to_csv(os.path.join(root_path, 'rbox.csv'), header=None, index=None)

# run cat <path/to/download.sh> | parallel -j 4

records_out = []
for image_path, data in tqdm(files.items()):
    txt_name = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    txt_path = os.path.join(txt_folder, txt_name)
    txt_path8 = os.path.join(txt_folder8, txt_name)

    # save to txt
    pd.DataFrame.from_records(data).to_csv(txt_path, header=None, index=None)
    pd.DataFrame.from_records(data).to_csv(txt_path8, header=None, index=None)
    records_out.append([image_path, txt_path])

# East training input csv
pd.DataFrame.from_records(records_out).to_csv(east_csv_path, header=None, index=None)
