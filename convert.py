# coding:utf-8
import argparse
import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, choices=['csvn', 'icdar'], required=True, help='Target data type')
# postprocess: to csvn
parser.add_argument('--input_folder', type=str, help='Input folder of groudtruth_boxes / eval results (txt)')
parser.add_argument('--groundtruth', action='store_true', default=False, help='Whether input is groundtruth')
parser.add_argument('--output_file', type=str, help='Output result csv file')
# preprocess: to icdar
parser.add_argument('--boxes_folder', type=str, help='Input folder of boxes, eight point format')

parser.set_defaults(
    input_folder='/data5/xin/ocr/output',
    ground_truth=False,
    output_file='/data5/xin/ocr/data_csv/result.csv',
    boxes_folder='/data5/xin/ocr/rot_boxes_v2_normal_with_class/boxes'
)

# parser.set_defaults(
#     input_folder='/data5/xin/ocr/rot_boxes_v2_normal_with_class/boxes/',
#     ground_truth=True,
#     output_file='/data5/xin/ocr/data_csv/groundtruth.csv',
#     boxes_folder='/data5/xin/ocr/rot_boxes_v2_normal_with_class/boxes/'
# )

args = parser.parse_args()

def valid(box_file, text_file):
    error = []
    for decode in ['ascii', 'utf-8', 'latin-1']:
        try:
            box_count = sum([1 for line in open(box_file, 'r', encoding=decode) if line.strip()])
            text_count = sum([1 for line in open(text_file, 'r', encoding=decode) if line.strip()])
            assert box_count == text_count
            return True
        except Exception as e:
            error.append(e)
            pass
    print(error)
    return False

def robust_decode(text_file):
    error = []
    for decode in ['ascii', 'utf-8', 'latin-1']:
        try:
            text = [item.strip().encode('utf-8') for item in open(text_file, 'r', encoding=decode).readlines()]
            return text
        except Exception as e:
            error.append(e)
            pass
    print(error)
    return None

def main():
    if args.type == 'csvn':
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
                    box_file, text_file = file, file.replace('/boxes', '/text')
                    if not valid(box_file, text_file):
                        print('==> Ignore {}'.format(box_file))
                        continue

                    # path, image_width, image_height, left, top, right, bottom, label
                    records.append([image_name, -1, -1, min_w, min_h, max_w, max_h, 'text'])
                else:
                    assert len(item) == 9
                    score = item[-1]
                    # path, timestamp, image_width, image_height, left, top, right, bottom, score, label
                    records.append([image_name, -1, -1, -1, min_w, min_h, max_w, max_h, score, 'text'])

        df = pd.DataFrame.from_records(records)
        df.to_csv(args.output_file, header=None, index=None)
        print('==> Save csv output to {}'.format(args.output_file))

    else: # 'icdar'
        '''
        Custom dataset folder:
        rot_boxes_v2_class/
            images/
                0.jpeg
                1.jpeg
            boxes/
                0.txt
                1.txt
            text/
                0.txt
                1.txt

        Save icdar format data (combination of boxes + text) to
            images/
                0.txt
                1.txt
        '''
        box_files = glob.glob(os.path.join(args.boxes_folder, '*.txt'))
        text_files = list(map(lambda x: x.replace('/boxes/', '/text/'), box_files))

        for idx, (box_file, text_file) in tqdm(enumerate(zip(box_files, text_files)), total=len(box_files)):
            if not valid(box_file, text_file):
                print('==> Ignore {}'.format(box_file))
                continue

            boxes = np.genfromtxt(box_file, delimiter=',').astype(np.float32)
            text = robust_decode(text_file)
            if text == None:
                print('==> Ignore {}'.format(box_file))
                continue

            records = []
            for box, label in zip(boxes, text):
                # x1, y1, x2, y2, x3, y3, x4, y4, label
                records.append(list(box) + [label])

            output_file = box_file.replace('/boxes/', '/images/')
            pd.DataFrame.from_records(records).to_csv(output_file, header=None, index=None)

if __name__ == '__main__':
    main()
