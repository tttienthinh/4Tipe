import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import Activation




class Convolutional:
    def __init__(self, kernel_size=3, nb_kernel=8, lr=0.1, activation=Activation.relu, d_activation=Activation.d_relu, mini=0, maxi=1):
        """
        Crée un layer de n neuronne connecté aux layer de input neuronnes
        """
        # (nb_kernel, kernel_size, kernel_size)
        self.kernels = np.random.rand(nb_kernel, kernel_size, kernel_size)*(maxi-mini)+mini
        self.kernel_size = kernel_size # matrice carrée de taille kernel_size*kernel_size
        self.nb_kernel = nb_kernel # nombre total de filtre
        self.lr = lr # learning rate

        # the name of the layer is 1
        # next one is 2 and previous 0
        self.predicted_output_ = 0
        self.predicted_output  = 0
        self.input_data = 0

        # Fonction d'activation
        self.activation = activation
        self.d_activation = d_activation

    def taille_sortie(self, img_size, kernel_size):
        # calcule la taille de sortie des images
        num_pixels = img_size-kernel_size+1
        return max(0, num_pixels)
        
    def convolve(self, imgs):
        k = self.kernel_size
        nb_img = img.shape[0]
        taille_sortie = self.taille_sortie(
            img_size=img.shape[1],
            kernel_size=k
        )
        # Initialisation des images de sorties
        convolved_imgs = np.zeros(shape=(self.nb_kernel*nb_img, taille_sortie, taille_sortie))
        
        for i_kernel in range(self.nb_kernel):
            kernel = self.kernels[i_kernel, :]
            for i_img in range(nb_img):
                img = imgs[i_img, :]
                # sur les lignes
                for i in range(taille_sortie):
                    # sur les colonnes
                    for j in range(taille_sortie):
                        # On extrait la sous matrice
                        mat = img[i:i+k, j:j+k]
                        # produit de Hadamard
                        convolved_imgs[i_img+i_kernel*self.nb_kernel, i, j] = np.sum(np.multiply(mat, kernel))
        return convolved_imgs

    def calculate(self, input_data): 
        # input est de taille (nb_donne, nb_images, hauteur, largeur)
        # On suppose (hauteur = largeur) : image carré
        """
        Calcule la sortie
        """
        nb_donne = len(input_data)
        y1 = np.array([
            self.convolve(input_data[i]) for i in range(nb_donne)
        ])
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output = z1
        return y1, z1

    def learn(self, e_2):
        """
        Permet de mettre à jour les poids weigth
        """
        e1 = e_2 / (self.kernel_size**2) * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0

