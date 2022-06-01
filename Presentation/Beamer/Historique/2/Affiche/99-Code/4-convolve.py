import numpy as np 
from PIL import Image, ImageOps
import matplotlib.pyplot as plt











kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
# convolve
def convolve(img):
    ks = kernel.shape[0]
    ts = img.shape[0]-ks+1
    # Initialisation de l'image
    c_img = np.zeros(shape=(ts,ts))
    for i in range(ts):
        for j in range(ts):
            # sous matrice
            m = img[i:i+ks, j:j+ks]
            # produit d'Hadamard
            c_img[i, j] = np.sum(
                m*kernel
            )
    return c_img