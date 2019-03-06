import os,sys
import numpy as np
from iou2 import get_iou
from operator import itemgetter
import logging
from tqdm import tqdm

IOU_THRESH=0.3
THRESH = 0.0
pred_file = sys.argv[1] if len(sys.argv) >= 2 else  '/data5/xin/object_dectection_hookit57/fixed.csv'
gt_file = sys.argv[2] if len(sys.argv) >= 3 else '/data5/xin/irv2_atrous/test.txt'

logging.warning('==> pred_file: {}'.format(pred_file))

gt = {}
#read in gt
with open(gt_file) as f:
    for line in f:
        line = line.rstrip()
        path,width,height,left,top,right,bottom,label = line.split(',')[:8]

        #if label != 'kingfisher_beer': continue

        left,top,right,bottom = float(left),float(top),float(right),float(bottom)

        if path in gt:
            gt[path].append([left,top,right,bottom,label])
        else:
            gt[path] = [[left,top,right,bottom,label]]
 

#read in preds
preds = {}
with open(pred_file) as f:
    for line in f:
        line = line.rstrip()
        path,timestamp,image_width,image_height,left,top,right,bottom,score,label = line.split(',')[:10]
        #path,timestamp,image_width,image_height,left,top,right,bottom,score,label,c_score = line.split(',')
        #path,image_width,image_height,left,top,right,bottom,label,score = line.split(',')

        left,top,right,bottom = float(left),float(top),float(right),float(bottom)
        score = float(score)   #TODO: Use c-score
        width = np.abs(right - left)
        height = np.abs(bottom - top)
        
        if score >= THRESH:
            if path in preds:
                preds[path].append([left,top,right,bottom,score,label])
            else:
                preds[path] = [[left,top,right,bottom,score,label]]
 


#first do recall
found = 0
total = 0
#for each image
total_preds = 0 
for path in tqdm(gt):

    current_gt = gt[path]
    total += len(current_gt)

    #print("%d gt" % len(current_gt))

    if path in preds:
        current_preds = preds[path]
    else:
        continue  #no preds for this image


    #print("%d preds" % len(current_preds))

    #for each ground truth in current image
    for g in current_gt:

        #print("     Current gt: ", g)

        #find pred with iou > IOU_THRESH
        total_preds += len(current_preds)
        for p in current_preds:
            #print(" current pred: ", p)
            #if the labels match
            if p[-1] == g[-1]:
                iou = get_iou(p,g)
                #print("iou", iou)
                if iou >= IOU_THRESH:
                    #print("match")
                    found += 1
                    break
       
#print('total_preds', total_preds)
#print("recall = %d/%d = %g" %(found,total,float(found)/total))
#exit()


for path in tqdm(preds):
    
    current_preds = preds[path]
    
    #should always have some gt for each image
    if path in gt:
        current_gt = gt[path]
    else:
        continue

    current_preds = sorted(current_preds,key=itemgetter(4),reverse=True) #sort by score

    for p in current_preds:
        correct = False
        max_iou = -1.0
        for g in current_gt:

            if p[-1] == g[-1]:
                iou = get_iou(p,g)

                #prediction p matches g ground truth
                #and is therefore correct
                if iou > max_iou:
                    max_iou = iou
                    max_g = g
        if max_iou >= IOU_THRESH: 
            current_gt.remove(max_g) 
            answer = 'yes'
        else:
            answer = 'no'

        print("%g,%s" %(p[-2],answer))

            
        



        
