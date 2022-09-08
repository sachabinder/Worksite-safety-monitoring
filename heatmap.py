
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
from PIL import Image
import os


#=================================== Image diffenrentiation
def difference_image(image1, image2):
    tab1 = np.asarray(image1)
    tab2 = np.asarray(image2)
    return Image.fromarray(tab1 - tab2)


def norm_image(img):
    tab = np.asarray(img)
    return np.linalg.norm(tab, ord=1)


def gaussian(x):
    return 255 * np.exp(-((x - 127.5) / 30) ** 2)


def is_same_cam(img1, img2, threshold):
    if img1.size == img2.size:
        I = difference_image(img1, img2)
        I = Image.eval(I, gaussian)
        return norm_image(I) < threshold
    else:
        return False


def image_differentiation(threshold, path):
    image_name_list = os.listdir(path)
    camera_list = []

    for imageName in image_name_list:
        img1 = Image.open(path + "/" + imageName)
        img1 = img1.convert('L')  # Greyscale conversion
        new_cam_necessary = True
        for camera in camera_list:
            img2 = Image.open(path + "/" + camera[0])
            img2 = img2.convert('L')
            if is_same_cam(img1, img2, threshold):
                camera.append(imageName)
                new_cam_necessary = False

        if new_cam_necessary:
            camera_list.append([imageName])

    return camera_list


#=================================== Heatmap

def similar_name(img_name1, img_name2):
    test_length = 15
    return img_name1[:test_length] == img_name2[:test_length]


def sort_images_names(path):    # only works for the training set
    image_name_list = os.listdir(path)
    camera_list = []
    for img_name1 in image_name_list:
        new_cam_necessary = True
        for camera in camera_list:
            img_name2 = camera[0]
            if similar_name(img_name1, img_name2):
                camera.append(img_name1)
                new_cam_necessary = False
                break

        if new_cam_necessary:
            camera_list.append([img_name1])

    return camera_list


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


def gradient(ratio):
    ratio = 1.2*ratio
    RGB = np.array([0, 0, 0])
    if ratio < 0.1:
        RGB[2] = 0.5 + 5*ratio
    elif ratio < 0.3:
        RGB[1] = (ratio-0.1)*5
        RGB[2] = 1
    elif ratio < 0.5:
        RGB[0] = (ratio-0.3)*5
        RGB[1] = 1
        RGB[2] = (0.5-ratio)*5
    elif ratio < 0.7:
        RGB[0] = 1
        RGB[1] = (0.7-ratio)*5
    elif ratio <0.8:
        RGB[0] = (0.8-ratio)*5
    else:
        RGB[0] = 127
    return 255*RGB


def add_gradient(heat_array):
    heat_array = heat_array/np.max(heat_array)
    w, h = np.shape(heat_array)
    heat_array_col = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            heat_array_col[i, j] = gradient(heat_array[i, j])
    return np.array(heat_array_col, np.int8)


def heatmap(json_path_lst, img_path, class_to_detect="People"):
    img = cv2.imread(img_path)  # Read image with cv2
    heat_array = np.zeros(img.shape[:2])

    for json_path in json_path_lst:
        boxes = get_bounding_box(json_path, class_to_detect)
        for [[x1, y1], [x2, y2]] in boxes:
            for x in range(x1, x2):
                for y in range(y1, y2):
                    heat_array[y, x] += 1

    heat_array = add_gradient(heat_array)
    heat_img = Image.fromarray(heat_array, mode='RGB')
    img = Image.open(img_path)
    new_image = Image.blend(heat_img, img, .25)
    new_image.show()



if __name__ == "__main__":

    path_img_train = "Detection_Train_Set/Detection_Train_Set_Img"
    path_json_train = "Detection_Train_Set/Detection_Train_Set_Json"
    path_img_test = "Detection_Test_Set/Detection_Test_Set_Img"
    path_json_test = "Detection_Test_Set/Detection_Test_Set_Json"

    #camera_lst = image_differentiation(50000, path_img)
    camera_lst = sort_images_names(path_img_train)

    for camera in camera_lst:
        json_path_list = [path_json_train + "/" + link + ".json" for link in camera]
        heatmap(json_path_list, path_img_train + "/" + camera[0])

    json_test_path = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg.json'
    img_test_path = 'Detection_Train_Set/Detection_Train_Set_Img/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg'
    # display_detection(img_test_path, json_test_path)
