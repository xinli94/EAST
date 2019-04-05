import cv2
import csv
import numpy as np
import json

from .vis_rbox import draw_rbox

image = 'ski.jpg'
east = 'ski_east.csv'
serving = 'ski_serving.csv'
ms = 'ski_ms.json'

def draw_rbox_csv(input_csv, im, offset=4, color=(255,0,0)):
    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        rboxes, labels = [], []
        for idx, row in enumerate(csv_reader):
            rbox = np.array([float(item) for item in row[offset:offset+8]]).reshape((4, 2)).astype(np.int32)
            label = row[offset+9]
            rboxes.append(rbox)
            labels.append(label)

    return draw_rbox(im, rboxes, labels, None, output_path=None, print_score=False, print_label=True, color=color)


def draw_rbox_json(input_json, im, color=(0,255,0)):
    with open(ms) as f:
        data = json.load(f)
        results = data['recognitionResult']['lines']
        rboxes, labels = [], []
        for res in results:
            words = res['words']
            for w in words:
                rboxes.append(np.array(w['boundingBox']).reshape((4, 2)).astype(np.int32))
                labels.append(w['text'])
        
    return draw_rbox(im, rboxes, labels, None, output_path=None, print_score=False, print_label=True, color=color)

def draw_bbox_csv(input_csv, im, offset=4, color=(0,0,255)):
    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        rboxes, labels = [], []
        for idx, row in enumerate(csv_reader):
            l,t,r,b = [float(item) for item in row[offset:offset+4]]
            rbox = np.array([[l,t], [r,t], [r,b], [l,b]]).astype(np.int32)
            label = row[offset+7]
            rboxes.append(rbox)
            labels.append(label)

    return draw_rbox(im, rboxes, labels, None, output_path=None, print_score=False, print_label=True, color=color)

im = cv2.imread(image) 
im_res = draw_rbox_csv(east, im)
im_res = draw_rbox_json(ms, im_res)
im_res = draw_bbox_csv(serving, im_res)

cv2.imwrite(image + '_benchmark.png', im_res)