import cv2
import glob
import time
import math
import numpy as np
import os
import pandas as pd
from PIL import Image, ImageDraw
import random
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms

random.seed(12345)

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_integer('write_images_count', 500, 'number of images to visualize')
tf.app.flags.DEFINE_float('score_threshold', 0.8, 'score threshold to use')
tf.app.flags.DEFINE_float('vis_score_threshold', 0.8, 'score threshold to use')
tf.app.flags.DEFINE_string('backbone', 'resnet_v1_50', 'backbone model to use')
tf.app.flags.DEFINE_boolean('vis_only', False, 'only visualize demo images')

import model
from icdar import restore_rectangle
from data_util import get_data

FLAGS = tf.app.flags.FLAGS
LIST_EXT = ['csv', 'txt']
IMAGES_EXT = ['jpg', 'png', 'jpeg', 'JPG']


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    mask = np.argsort(xy_text[:, 0])
    xy_text = xy_text[mask]
    # scores
    scores_filtered = score_map[score_map > score_map_thresh]
    scores_filtered = scores_filtered[mask]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer, None

    # here we filter some low score boxes by the average score map, this is different from the original paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    print('{} text boxes after nms'.format(boxes.shape[0]))

    return boxes, timer, scores_filtered


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def vis_score(score_map, im_fn, im_resized, ratio_h, ratio_w, score_map_thresh=0.8, pixel_size=4.0):
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]

    activation_pixels = np.where(score_map > score_map_thresh)

    im = Image.fromarray(im_resized)
    draw = ImageDraw.Draw(im)
    # psx, psy = 0.5 / ratio_h, 0.5 / ratio_w
    psx = psy = 0.5
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        # px = (j + 0.5) * pixel_size / ratio_h
        # py = (i + 0.5) * pixel_size / ratio_w
        px = (j + psx) * pixel_size
        py = (i + psy) * pixel_size
        line_width, line_color = 1, 'red'
        draw.line([(px - psx * pixel_size, py - psy * pixel_size),
                   (px + psx * pixel_size, py - psy * pixel_size),
                   (px + psx * pixel_size, py + psy * pixel_size),
                   (px - psx * pixel_size, py + psy * pixel_size),
                   (px - psx * pixel_size, py - psy * pixel_size)],
                  width=line_width, fill=line_color)
    image_path = os.path.join(FLAGS.output_dir, os.path.splitext(os.path.basename(im_fn))[0] + '_act.jpg')
    im.save(image_path)


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False, backbone=FLAGS.backbone)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        image_count = 0

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            except:
                model_path = FLAGS.checkpoint_path
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list, _ = get_data(FLAGS.test_data_path)
            random.shuffle(im_fn_list)
            total = len(im_fn_list)
            for idx, im_fn in enumerate(im_fn_list):
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})

                timer['net'] = time.time() - start

                boxes, timer, scores_filtered = detect(
                    score_map=score,
                    geo_map=geometry,
                    timer=timer,
                    score_map_thresh=FLAGS.score_threshold
                )

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time

                print('[{}/{} ({:.2f}%)] {} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms, [timing] {:.2f}s'.format(
                    idx+1, total, 100.0 * (idx+1) / total, im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000, duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box, cur_score in zip(boxes, scores_filtered):
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], cur_score
                            ))
                            if cur_score > FLAGS.vis_score_threshold:
                                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 0, 0), thickness=2)

                if image_count < FLAGS.write_images_count:
                    image_count += 1
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
                    vis_score(score, im_fn, im_resized, ratio_h, ratio_w, score_map_thresh=FLAGS.score_threshold)
                elif FLAGS.vis_only:
                    print('==> Visualization only')
                    return

if __name__ == '__main__':
    tf.app.run()
