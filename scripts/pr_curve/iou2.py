#box_1 = {"x1":0,"x2":100,"y1":0,"y2":100}
#box_2 = {"x1":50,"x2":200,"y1":50,"y2":200}

from shapely.geometry import Polygon
import numpy as np

def get_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float in [0, 1]
    """

    bb1 = {'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2': box1[3]}
    bb2 = {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]}

    if bb1['x1'] >= bb1['x2']: return 0.0
    if bb1['y1'] >= bb1['y2']: return 0.0
    if bb2['x1'] >= bb2['x2']: return 0.0
    if bb2['y1'] >= bb2['y2']: return 0.0

    # determine the coordinates of the intersection rectangle

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou



def get_iomin(box1,box2):
    """
    Calculate the Intersection over the area over the smaller bbox.
    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    Returns
    -------
    float
        in [0, 1]
    """

    bb1 = {'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2': box1[3]}
    bb2 = {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]}


    if bb1['x1'] >= bb1['x2']:
        return 0.0
    if bb1['y1'] >= bb1['y2']:
        return 0.0
    if bb2['x1'] >= bb2['x2']:
        return 0.0
    if bb2['y1'] >= bb2['y2']:
        return 0.0


    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    min_area = min(bb1_area,bb2_area)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iomin = intersection_area / float(min_area)
    assert iomin >= 0.0
    assert iomin <= 1.0
    return iomin

#print get_iou(box_1,box_2)
#print get_iomin(box_1,box_2)


def get_iou_rbox(box1, box2):
    poly1 = Polygon(np.array(box1[:8]).reshape((4,2)).tolist())
    poly2 = Polygon(np.array(box2[:8]).reshape((4,2)).tolist())

    return poly1.intersection(poly2).area / poly1.union(poly2).area


