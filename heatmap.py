from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


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
        im = difference_image(img1, img2)
        im = Image.eval(im, gaussian)
        return norm_image(im) < threshold
    else:
        return False


def image_differentiation(threshold, path):
    camera_list = []
    image_name_list = os.listdir(path)

    for image_name in image_name_list:
        img1 = Image.open(path + image_name).convert('L')  # Opening and greyscale conversion
        new_cam_necessary = True

        for camera in camera_list:
            img2 = Image.open(path + camera[0]).convert('L')
            if is_same_cam(img1, img2, threshold):
                camera.append(image_name)
                new_cam_necessary = False

        if new_cam_necessary:
            camera_list.append([image_name])

    return camera_list


if __name__ == "__main__":
    path = "Detection_Test_Set/Detection_Test_Set_Img/"
    threshold = 55000

    CameraList = image_differentiation(threshold, path)
    for camera in CameraList:
        k = 1
        for image in camera:
            img = Image.open(path + image)
            if k < 25:
                plt.subplot(5, 5, k)
                plt.imshow(img)
            k += 1

        plt.show()
