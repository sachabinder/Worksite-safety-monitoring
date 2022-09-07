from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Différenciation d'images :
def echelles_de_gris(img):
    """Renvoie une image après l'avoir converti en échelle de gris"""
    return img.convert('L')

def differenceImage(image1, image2):
    """ Renvoie une image égal à la différence entre image1 et image2"""
    tab1 = np.asarray(image1)
    tab2 = np.asarray(image2)

    return Image.fromarray(tab1 - tab2)

def normImage(img):
    """Renvoie la norme d'un tableau représentant une image en échelle de gris"""
    tab = np.asarray(img)
    return np.linalg.norm(tab, ord=1)

def sigmoid(x):
    if x/255 < 0.5:
        return 255*(2*(x/255)**2)
    else:
        return 255*(-2*(x/255)**2 + 4*(x/255) - 1)

def gaussienne(x):
    return 255*np.exp(-((x-127.5)/30)**2)

def transformationSigmoidImage(img):
    """
    Renvoie une image en échelle de gris après amplification par une fonction sigmoïde.
    img : image en échelle de gris
    """
    return Image.eval(img,sigmoid)

def transformationClocheImage(img):
    """
    Renvoie une image en échelle de gris après amplification par une fonction sigmoïde.
    img : image en échelle de gris
    """
    return Image.eval(img, gaussienne)
#-------------------------------------
def displayImage(img, grey = True):
    """Affiche une image. Si on veut l'afficher en noir et blanc, il faut changer le terme 'grey' en false"""
    if grey:
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)
    plt.show()

def isSameCam(img1, img2,threshold):
    """Vérifie si deux images sont suffisamment similaires pour venir de la même caméra."""
    if img1.width == img2.width and img1.height == img2.height:
        I = differenceImage(img1, img2)
        I = transformationClocheImage(I)
        return normImage(I)<threshold
    else:
        return False

# --------------------------------------------------------------------------------

path = "Detection_Test_Set/Detection_Test_Set_Img"
ImageNameList = os.listdir(path)
CameraList = []
threshold = 50000


for imageName in ImageNameList:
    img1 = echelles_de_gris(Image.open(path + "/" + imageName))
    NewCamNecessary = True
    for camera in CameraList:
        img2 = echelles_de_gris(Image.open(path + "/" + camera[0]))
        if isSameCam(img1,img2,threshold):
            camera.append(imageName)
            NewCamNecessary = False

    if NewCamNecessary:
        CameraList.append([imageName])
