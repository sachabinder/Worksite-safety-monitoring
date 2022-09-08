import numpy as np
from typing import List, Dict, Tuple
from dataset import data_set_builder

dataset = data_set_builder()
#print(dataset[0]['img_file_name'])
#print(dataset[0]['img'])
#print(dataset[0]['boxes_labelled'])
#print(dataset[0]['target_file_name'])


### Intersection over Union 

def compare_precesion_recall(precision,recall):
    return 2 * (precision * recall)/(precision + recall)

thersholds = np.linspace(0.1,1,10)
test_thers= 0.5
#print(thersholds)

data0 = dataset[0]

def format_json(data): #met sous la bonne forme les coordonées de json pour IoU
    boxes_coordonates=[]
    for k in range(len(data['boxes_labelled']['boxes'])):
        coordonnate = data['boxes_labelled']['boxes'][k]
        boxes_coordonates.append(coordonnate)
    return boxes_coordonates

#def format_img_model(data): #met sous la bonne forme les coordonées venant du modee pour IoU
    #boxes_coordonates=[]
    #for k in range(len(boxes)):
     #   coordonate = ## 
    #    boxes_coordonates.append(coordonnate)
    #return 

## le format xyxy permet d'avoir les coordonées du point en bas à droite en premier puis celui en haut à gauche


def IoU(data): # gt_box = json, pred_box = from model
    liste_iou=[]
    N=len(format_json(data))
    for k in range(N):
        gt_box = format_json(data)[k]
        #gt_box , pred_box = format_json(data) , format_img_model(data)
        pred_box = format_json(dataset[2])[k]
        #print(gt_box[0])
        inter_box_top_left = [min(gt_box[2], pred_box[2]),min(gt_box[3], pred_box[3])]
        inter_box_bottom_right = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]

        inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
        inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

        intersection = inter_box_w * inter_box_h
        union = (gt_box[0]-gt_box[2])*(gt_box[1]-gt_box[3]) + (pred_box[0]-pred_box[2])*(pred_box[1]-pred_box[3]) 
        
        iou = intersection / union 
        print(gt_box,pred_box)
        print(intersection,union)
        liste_iou.append(iou)
    return liste_iou

#print(IoU(data0))


def total_accepted(thershold,data):
    list_iou = IoU(data)
    tot = 0
    for iou in list_iou:
        if iou >= thershold :
            tot+=1
    return tot, len(list_iou)

def precision(thershold):
    tot_accepted = 0
    tot=0 
    for data in dataset :
        tot_accept_data , tot_data = total_accepted(thershold,data)
        tot += tot_data 
        tot_accepted += tot_accept_data
    return tot_accepted/tot 





