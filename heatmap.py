
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

def add_zeros(path, file="jpg"):
    if file == "jpg":
        if path[-8] == 'e':
            path = path[:-7] + '0' + path[-7:]
        if path[-7] == 'e':
            path = path[:-6] + "00" + path[-6:]
    if file == "json":
        if path[-13] == 'e':
            path = path[:-12] + '0' + path[-12:]
        if path[-12] == 'e':
            path = path[:-11] + "00" + path[-11:]
    return path


def camera_list_train_set():
    ''' Create the list of the different cameras only for training set
        (The training set is a little bit messy, that's why this function is terrible)
    '''
    path_img_train = "Detection_Train_Set/Detection_Train_Set_Img"
    path_json_train = "Detection_Train_Set/Detection_Train_Set_Json"

    first_names = ["Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg",
                   "Batch2__Devisubox2_06frame0053.jpg",
                   "Batch2__Devisubox2_06frame5209.jpg",
                   "Batch2__Marseille_01frame0530.jpg",
                   "Batch2__Marseille_01frame0707.jpg",
                   "Batch2__Marseille_01frame0911.jpg",
                   "Batch2__Marseille_01frame1057.jpg",
                   "Batch2__Marseille_01frame1168.jpg",
                   "Batch2__Marseille_01frame1334.jpg",
                   "Batch2__Nouveau_campus_03frame0660.jpg",
                   "Batch2__Roissy_02frame0911.jpg"
                   ]

    corresponding_camera = [0, 1, 2, 3, 4, 5, 4, 5, 4, 6, 7]

    camera_lst = [[], [], [], [], [], [], [], []]
    index = 0
    image_name_list = os.listdir(path_img_train)

    for image_name in image_name_list:
        path_img = path_img_train + "/" + image_name
        path_json = path_json_train + "/" + image_name + ".json"
        os.rename(path_img, add_zeros(path_img))
        os.rename(path_json, add_zeros(path_json, file="json"))
        if index < 11 and image_name == first_names[index]:
            index += 1
        camera_lst[corresponding_camera[index-1]].append(image_name)
    return camera_lst


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
    """ return the RGB value to give for a gradient with ratio in [0,1]
    """
    ratio = 1.2*ratio+0.1    # shift of the gradient
    RGB = np.array([0., 0., 0.])
    if ratio < 0.2:
        RGB[2] = 5*ratio
    elif ratio < 0.4:
        RGB[1] = (ratio-0.2)*5
        RGB[2] = 1
    elif ratio < 0.6:
        RGB[0] = (ratio-0.4)*5
        RGB[1] = 1
        RGB[2] = (0.6-ratio)*5
    elif ratio < 0.8:
        RGB[0] = 1
        RGB[1] = (0.8-ratio)*5
    elif ratio < 0.9:
        RGB[0] = (1-ratio)*5
    else:
        RGB[0] = 0.5
    return 255*RGB


def add_gradient(heat_array):
    """ transform a array in an array with rgb values with a gradient
    """
    heat_array = heat_array/np.max(heat_array)
    w, h = np.shape(heat_array)
    heat_array_col = np.zeros((w, h, 3))
    for i in range(w):
        for j in range(h):
            heat_array_col[i, j] = gradient(heat_array[i, j])
    return np.array(heat_array_col, np.int8)


def heatmap(json_path_lst, img_path, class_to_detect="People"):
    img = cv2.imread(img_path)  # Read image with cv2
    w, h = img.shape[:2]
    heat_array = np.zeros((w, h))

    for json_path in json_path_lst:
        boxes = get_bounding_box(json_path, class_to_detect)
        for [[x1, y1], [x2, y2]] in boxes:
            dx = x2-x1
            dy = y2-y1
            ratio = 1.5
            m_x = (x1+x2)/2
            m_y = (y1+y2)/2
            x3 = max(0, int(x1-ratio*dx))
            x4 = min(h, int(x2+ratio*dx))
            y3 = max(0, int(y1 - ratio*dy))
            y4 = min(w, int(y2 + ratio*dy))
            for x in range(x3, x4):
                for y in range(y3, y4):
                    dist2 = ((x-m_x)/dx)**2+((y-m_y)/dy)**2
                    if ((x-m_x)/dx)**2+((y-m_y)/dy)**2 < ratio**2:
                        heat_array[y, x] += (ratio-dist2**0.5)/(0.2+dist2)

    heat_array = add_gradient(heat_array)
    heat_img = Image.fromarray(heat_array, mode='RGB')
    img = Image.open(img_path)
    new_image = Image.blend(heat_img, img, .6)
    new_image.show()
    return new_image


def heatmap_demo():
    camera_lst = camera_list_train_set()

    path_img_train = "Detection_Train_Set/Detection_Train_Set_Img"
    path_json_train = "Detection_Train_Set/Detection_Train_Set_Json"
    path_img_test = "Detection_Test_Set/Detection_Test_Set_Img"
    path_json_test = "Detection_Test_Set/Detection_Test_Set_Json"

    i = 0
    for camera in camera_lst:
        json_path_list = [path_json_train + "/" + link + ".json" for link in camera]
        image = heatmap(json_path_list, path_img_train + "/" + camera[0])
        image.save("Detection_Train_Set/heatmaps/heatmap_" + str(i) + ".png", 'png')
        i += 1


def detection_demo():

    json_test_path = 'Detection_Train_Set/Detection_Train_Set_Json/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg.json'
    img_test_path = 'Detection_Train_Set/Detection_Train_Set_Img/Batch2__BioSAV_BIofiltration_18mois_05frame3049.jpg'

    display_detection(img_test_path, json_test_path)
