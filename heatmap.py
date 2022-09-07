import matplotlib.pyplot as plt
import cv2
import json
import numpy as np


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


json_test_path = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg.json'
img_test_path = 'Detection_Train_Set/Detection_Train_Set_Img/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg'

test_display(img_test_path, json_test_path)