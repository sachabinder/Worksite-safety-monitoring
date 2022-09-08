import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import pandas as pd
import seaborn as sn


def open_json(json_path):

    with open(json_path) as json_file:
        return json.load(json_file)


def extract_polygons(json_path, class_to_detect="People"):

    data = open_json(json_path)

    if "objects" in data:   # For json in the given dataset
        return [np.array(item["points"]["exterior"], np.int32)
                for item in data["objects"]
                if item["classTitle"] == class_to_detect]
    else:                   # For json build by the program
        return [np.reshape(np.array(item["boxes"], np.int32), (2, 2))
                for item in data["boxes_labelled"]
                if item["label"] == class_to_detect]


def test_display(img_path, json_path, class_to_detect="People", line_th=2):
    """ Draw polygons around the class to detect
        NB : for 'People' the polygons are just lines
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    polygons = extract_polygons(json_path, class_to_detect)

    for polygon in polygons :
        cv2.polylines(img, [polygon], isClosed=True, color=(255, 0, 0), thickness=line_th)

    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def heatmap(json_path_lst, img_path, class_to_detect="People", resize_ratio=10):
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    [w, h] = img.shape[:2]
    w_heat = w//resize_ratio
    h_heat = h//resize_ratio
    heat_array = np.zeros((w_heat, h_heat))

    for json_path in json_path_lst:
        polygons = extract_polygons(json_path,class_to_detect)
        for polygon in polygons:
            x1 = min(polygon[0][0], polygon[1][0])
            x2 = max(polygon[0][0], polygon[1][0])
            y1 = min(polygon[0][1], polygon[1][1])
            y2 = max(polygon[0][1], polygon[1][1])
            for x in range(x1//resize_ratio, x2//resize_ratio):
                for y in range(y1//resize_ratio, y2//resize_ratio):
                    heat_array[y, x] += 1

    df = pd.DataFrame(heat_array, columns=None)
    sn.heatmap(df, xticklabels=False, yticklabels=False, cbar=False)
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.show()


# For test


json_test_path = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame3059.jpg.json'
img_test_path = 'Detection_Train_Set/Detection_Train_Set_Img/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg'
json_begining = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame'

json_test_path_list = []
for frame in range(3059, 3540, 5):
    if frame != 3289:
        json_test_path_list.append(json_begining + str(frame) + ".jpg.json")

#test_display(img_test_path, json_test_path)
heatmap(json_test_path_list, img_test_path)