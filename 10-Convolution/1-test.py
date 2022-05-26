import numpy as np 
from PIL import Image, ImageOps
import matplotlib.pyplot as plt





# kernel
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125],
    [0.0625, 0.125, 0.0625]
])

outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])


# convolve
def taille_sortie(img_size, kernel_size):
    # calcule la taille de sortie des images
    num_pixels = img_size-kernel_size+1
    return max(0, num_pixels)


def convolve(img, kernel):
    k = kernel.shape[0]
    ts = taille_sortie(
        img_size=img.shape[0],
        kernel_size=k
    )
    # Initialisation de l'image
    convolved_img = np.zeros(shape=(ts, ts))
    
    # sur les lignes
    for i in range(ts):
        # sur les colonnes
        for j in range(ts):
            # On extrait la sous matrice
            mat = img[i:i+k, j:j+k]
            # produit de Hadamard
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))
    return convolved_img



def negative_to_zero(img: np.array) -> np.array:
    img = img.copy()
    img[img < 0] = 0
    return img
    
def positive_to_one(img: np.array) -> np.array:
    img = img.copy()
    img = 1- np.exp(-img/100) 	
    return img

def plot_image(img: np.array):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    
def plot_two_images(img1: np.array, img2: np.array, gray=True):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    if gray:
        ax[0].imshow(img1, cmap='gray')
        ax[1].imshow(img2, cmap='gray')
    else:
        ax[0].imshow(img1)
        ax[1].imshow(img2, cmap='gray')


outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])


for animal, i in [("cat", 10), ("dog", 13)]:

    img_c = Image.open(f'TensorFlow/data/train/{animal}/{i}.jpg')
    img_c = img_c.resize(size=(224, 224))
    img = ImageOps.grayscale(img_c)
    img_outlined = convolve(img=np.array(img), kernel=outline)
    img_outlined = negative_to_zero(img=img_outlined)
    img_outlined = positive_to_one(img=img_outlined)
    plot_two_images(
        img1=img_c, 
        img2=img_outlined,
        gray=False
    )
    plt.savefig(f"1-image_{animal}_{i}.jpg", bbox_inches='tight', pad_inches=0)
