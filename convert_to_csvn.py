import argparse
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--args.input_folder', type=str, help='Input folder')
parser.add_argument('--groundtruth', action='store_true', help='Whether input is groundtruth')
parser.add_argument('--output_file', type=str, help='Output file')

parser.set_defaults(
    input_folder='/data5/xin/ocr/output',
    ground_truth=False,
    output_file='/data5/xin/ocr/data_csv/result.csv'
)

# parser.set_defaults(
#     input_folder='/data5/xin/ocr/rot_boxes_v2_class/boxes/',
#     ground_truth=True,
#     output_file='/data5/xin/ocr/data_csv/groundtruth.csv'
# )

args = parser.parse_args()

files = glob.glob(os.path.join(args.input_folder, '*.txt'))
records = []
for file in tqdm(files):
    image_name = os.path.splitext(os.path.basename(file))[0] + '.jpeg'

    data = np.genfromtxt(file, delimiter=',')
    for idx, item in enumerate(data):
        box = item[:8].reshape(4,2)
        min_h, max_h = box[:,1].min(), box[:,1].max()
        min_w, max_w = box[:,0].min(), box[:,0].max()
        if args.ground_truth:
            # path,image_width,image_height,left,top,right,bottom,label
            records.append([image_name, -1, -1, min_w, min_h, max_w, max_h, 'text'])
        else:
            assert len(item) == 9
            score = item[-1]
            # path,timestamp,image_width,image_height,left,top,right,bottom,score,label
            records.append([image_name, -1, -1, -1, min_w, min_h, max_w, max_h, score, 'text'])

df = pd.DataFrame.from_records(records)
df.to_csv(args.output_file, header=None, index=None)
