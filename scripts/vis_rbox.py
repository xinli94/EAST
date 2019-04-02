import collections
import csv
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def draw_rbox(im, rboxes, words, scores, output_path, print_score, print_label, color=(255, 0, 0), line_thickness=1):
    img = im.copy()

    for idx, rbox in enumerate(rboxes):
        rbox = rbox.astype(np.int32).reshape((4, 2))

        # visualize rbox
        cv2.polylines(img, [rbox], True, color=color, thickness=line_thickness)

        # text_x, text_y = rbox[0, 0], rbox[0, 1] + 5

        left, top = rbox[:, 0].min(), rbox[:, 1].min()
        left1, top1 = max(left - 1, 0), max(top - 1, 0)
        text_x = int(left1)
        text_y = int(max(top1 - 3, 0))

        if print_label:
            if print_score:
                # visualize score
                cv2.putText(img, words[idx] + ": " + '{:.2f}'.format(scores[idx]), (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness=1, lineType=cv2.LINE_AA)
            else:
                # visualize label
                cv2.putText(img, words[idx], (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, thickness=1, lineType=cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, img)
    return img


def draw_rbox_from_csv(input_csv, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')

        files = collections.defaultdict(lambda: collections.defaultdict(list))
        for idx, row in enumerate(csv_reader):
            image_path = row[0]
            rbox = np.array([float(item) for item in row[1:9]]).reshape((4, 2)).astype(np.int32)
            label = row[-1]
            files[image_path]['rboxes'].append(rbox)
            files[image_path]['labels'].append(label)

        for image_path, data in tqdm(files.items()):
            records = []
            rboxes, labels = data['rboxes'], data['labels']

            im = cv2.imread(image_path)
            output_path = os.path.join(output_folder, os.path.basename(image_path) + '.out.png')

            draw_rbox(im, rboxes, labels, None, output_path, print_score=False, print_label=True)


if __name__ == '__main__':
    input_csv = '/data5/xin/MSRA-TD500/msra_td500.csv'
    output_folder = './vis/'

    draw_rbox_from_csv(input_csv, output_folder)

