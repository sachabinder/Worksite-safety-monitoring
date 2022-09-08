from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


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


# ===============================================
def imageDifDBScan(threshold, path):
    ImageNameList = os.listdir(path)
    CameraList = []

    while len(ImageNameList) > 0:
        print(len(ImageNameList))
        img0 = ImageNameList.pop(0)
        neighbours = [img0]
        newCamera = [img0]

        while len(neighbours) > 0:
            print("    ", len(neighbours))
            name = neighbours.pop(0)
            img1 = Image.open(path + name).convert('L')

            for imageName in ImageNameList:
                img2 = Image.open(path + imageName).convert('L')
                if isSameCam(img1, img2, threshold):
                    neighbours.append(imageName)
                    newCamera.append(imageName)
                    ImageNameList.remove(imageName)

        CameraList.append(newCamera)

    return CameraList


if __name__ == "__main__":
    path = "Detection_Test_Set/Detection_Test_Set_Img/"
    CameraList = imageDifDBScan(45000, path)
    for camera in CameraList:
        k = 1
        for image in camera:
            img = Image.open(path + image)
            if k < 26:
                plt.subplot(5, 5, k)
                plt.imshow(img)
            k += 1

        plt.show()