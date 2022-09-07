import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import pandas as pd
import seaborn as sn


def open_json(json_path):
    with open(json_path) as json_file:
        return json.load(json_file)


def test_display(img_path, json_path, class_to_detect="People", line_th=2): # draw polygons around the class to detect
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    data = open_json(json_path)

    for entity in data["objects"]:
        if entity["classTitle"] == class_to_detect:
            pts = np.array(entity["points"]["exterior"],np.int32)
            pts.reshape((-1,1,2))
            cv2.polylines(img, [pts], isClosed=True, color=(255,0,0), thickness=line_th)

    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def heatmap(json_path_lst, img_path, class_to_detect="People", coeff=10):
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    [w, h] = img.shape[:2]
    w_heat = w//coeff
    h_heat = h//coeff
    heat_array = np.zeros((w_heat, h_heat))
    for json_path in json_path_lst:
        data = open_json(json_path)
        for entity in data["objects"] :
            if entity["classTitle"] == class_to_detect:
                pts = entity["points"]["exterior"]
                x1 = min(pts[0][0], pts[1][0])
                x2 = max(pts[0][0], pts[1][0])
                y1 = min(pts[0][1], pts[1][1])
                y2 = max(pts[0][1], pts[1][1])
                norm = (x2-x1)*(y2-y1)
                for x in range(x1//coeff, x2//coeff):
                    for y in range(y1//coeff, y2//coeff):
                        heat_array[y, x] += 1
    df = pd.DataFrame(heat_array, columns=None)
    sn.heatmap(df, xticklabels=False, yticklabels=False, cbar=False)
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.show()


json_test_path = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg.json'
img_test_path = 'Detection_Train_Set/Detection_Train_Set_Img/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg'


#test_display(img_test_path, json_test_path)
heatmap([json_test_path],img_test_path)