from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    return np.linalg.norm(tab, ord=2)

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
#------------------------------------- Affichage d'image
def displayImage(img, grey = True):
    """Affiche une image. Si on veut l'afficher en noir et blanc, il faut changer le terme 'grey' en false"""
    if grey:
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)
    plt.show()

# --------------------------------------------------------------------------------
img1 = Image.open("Detection_Test_Set/Detection_Test_Set_Img/2019-10-17-13-30-11.jpg")
img1 = echelles_de_gris(img1)

img2 = Image.open("Detection_Test_Set/Detection_Test_Set_Img/2019-10-17-13-42-19.jpg")
img2 = echelles_de_gris(img2)

img3 = Image.open("Detection_Test_Set/Detection_Test_Set_Img/2019-10-17-15-19-05.jpg")
img3 = echelles_de_gris(img3)

img4 = Image.open("Detection_Test_Set/Detection_Test_Set_Img/2020-07-15-11-36-02.jpg")
img4 = echelles_de_gris(img4)


#Comparaison image :
I1 = img1
I2 = img2
I = differenceImage(I1, I2)
I = transformationClocheImage(I)

displayImage(I)
print(normImage(I))
