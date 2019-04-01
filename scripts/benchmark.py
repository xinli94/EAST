import argparse
import collections
import csv
import cv2
import numpy as np
import os
from tqdm import tqdm

from vis_rbox import draw_rbox

def draw_helper(files, color, keep=False):
    for image_path, data in tqdm(files.items()):
        records = []
        rboxes, labels = data['rboxes'], data['labels']

        output_path = os.path.join(args.output_folder, os.path.basename(image_path) + '.out.png')
        if os.path.isfile(output_path) and keep:
            im = cv2.imread(output_path)
        else:
            im = cv2.imread(image_path)

        draw_rbox(im, rboxes, labels, None, output_path, print_score=False, print_label=True, color=color, line_thickness=2)


def draw_rbox_csv(input_csv, offset=4, color=(255,0,0), keep=False):
    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        files = collections.defaultdict(lambda: collections.defaultdict(list))
        for idx, row in enumerate(csv_reader):
            image_path = row[0]
            rbox = np.array([float(item) for item in row[offset:offset+8]]).reshape((4, 2)).astype(np.int32)
            label = row[offset+9]
            files[image_path]['rboxes'].append(rbox)
            files[image_path]['labels'].append(label)

    draw_helper(files, color, keep=keep)


def draw_bbox_csv(input_csv, offset=4, color=(0,0,255), keep=False):
    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        files = collections.defaultdict(lambda: collections.defaultdict(list))
        for idx, row in enumerate(csv_reader):
            image_path = row[0]
            l, t, r, b = [float(item) for item in row[offset:offset+4]]
            rbox = np.array([[l,t], [r,t], [r,b], [l,b]]).astype(np.int32)
            label = row[offset+7]
            files[image_path]['rboxes'].append(rbox)
            files[image_path]['labels'].append(label)

    draw_helper(files, color, keep=keep)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--east_csv', type=str, default='/data5/xin/ocr/bad_examples/res101_synthtext_cocotext_0319_literate_v4/output_rbox.txt')
    parser.add_argument('--serving_csv', type=str, default='/data5/xin/ocr/bad_examples/serving/output.txt')
    parser.add_argument('--output_folder', type=str, default='tmp')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    draw_rbox_csv(args.east_csv)
    draw_bbox_csv(args.serving_csv, keep=True)
