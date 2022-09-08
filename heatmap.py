
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import pandas as pd
import seaborn as sn
from PIL import Image
import os


#=================================== Image diffenrentiation
def differenceImage(image1, image2):
    tab1 = np.asarray(image1)
    tab2 = np.asarray(image2)
    return Image.fromarray(tab1 - tab2)


def normImage(img):
    tab = np.asarray(img)
    return np.linalg.norm(tab, ord=1)


def gaussienne(x):
    return 255 * np.exp(-((x - 127.5) / 30) ** 2)


def isSameCam(img1, img2, threshold):
    if img1.size == img2.size:
        I = differenceImage(img1, img2)
        I = Image.eval(I, gaussienne)
        return normImage(I) < threshold
    else:
        return False

# -----------
def imageDifferentiation(threshold, path):
    ImageNameList = os.listdir(path)
    CameraList = []

    for imageName in ImageNameList:
        img1 = Image.open(path + "/" + imageName)
        img1 = img1.convert('L')  # Greyscale conversion
        NewCamNecessary = True
        for camera in CameraList:
            img2 = Image.open(path + "/" + camera[0])
            img2 = img2.convert('L')
            if isSameCam(img1, img2, threshold):
                camera.append(imageName)
                NewCamNecessary = False

        if NewCamNecessary:
            CameraList.append([imageName])

    return CameraList



#=================================== Heatmap
def open_json(json_path):

    with open(json_path) as json_file:
        return json.load(json_file)


def get_polygons(json_path, class_to_detect="People"):

    data = open_json(json_path)
    polygons = []

    for item in data["objects"] :
        if item["classTitle"] == class_to_detect:
            if item["geometryType"] == "polygon":
                polygons.append(np.array(item["points"]["exterior"], np.int32))
            elif item["geometryType"] == "rectangle":
                [[x1, y1], [x2, y2]] = item["points"]["exterior"]
                polygons.append(np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], np.int32))

    return polygons


def get_bounding_box(json_path, class_to_detect='People'):

    data = open_json(json_path)
    boxes = []

    for item in data["objects"]:
        if item["classTitle"] == class_to_detect:
            if item["geometryType"] == "polygon":
                polygon = np.array(item["points"]["exterior"], np.int32)
                boxes.append([[min(polygon[:, 0]), min(polygon[:,1])], [min(polygon[:, 0]), min(polygon[:,1])]])

            elif item["geometryType"] == "rectangle":
                boxes.append(item["points"]["exterior"])
    return boxes


def display_detection(img_path, json_path, class_to_detect="People", line_th=2):
    """ Draw polygons around the class to detect
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    polygons = get_polygons(json_path, class_to_detect)

    for polygon in polygons:
        cv2.polylines(img, [polygon], isClosed=True, color=(255, 0, 0), thickness=line_th)

    plt.figure(figsize=(20, 30))  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def heatmap(json_path_lst, img_path, class_to_detect="People", resize_ratio=3):
    img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    [w, h] = img.shape[:2]
    heat_array = np.zeros((w//resize_ratio, h//resize_ratio))

    for json_path in json_path_lst:
        boxes = get_bounding_box(json_path, class_to_detect)
        for [[x1, y1], [x2, y2]] in boxes:
            for x in range(x1//resize_ratio, x2//resize_ratio):
                for y in range(y1//resize_ratio, y2//resize_ratio):
                    heat_array[y, x] += 1

    df = pd.DataFrame(heat_array, columns=None)
    sn.heatmap(df, xticklabels=False, yticklabels=False, cbar=False)
    plt.pcolor(df)
    plt.show()


if __name__ =="__main__":
    json_test_path = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame3059.jpg.json'
    img_test_path = 'Detection_Train_Set/Detection_Train_Set_Img/Batch2__BioSAV_BIofiltration_18mois_05frame3059.jpg'
    json_begining = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame'

    json_test_path_list = []
    for frame in range(3059, 3540, 5):
        if frame != 3289:
            json_test_path_list.append(json_begining + str(frame) + ".jpg.json")

    display_detection(img_test_path, json_test_path)
    heatmap(json_test_path_list, img_test_path)
  
    #----------
    imageDifferentiation(50000, "Detection_Test_Set/Detection_Test_Set_Img")
    