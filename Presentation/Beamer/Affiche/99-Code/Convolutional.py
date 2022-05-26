from tensorflow.keras.datasets import fashion_mnist, mnist # Seulement pour importer les images
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Model import Model, ModelClassification
from Layer import Layer, LayerOptimizer
from Activation import *
import Activation
from Loss import *




class Convolutional(Layer):
    """
    On hérite de la class Layer, 
    Au lieu de prendre la representation 
        de kernels appliqués aux images.

    Je représente les kernels sous la forme d'un Layer avec : 
        - input_n = kernel_size**2
        - output_n = nb_kernel
    """
    def __init__(self, img_size:np.ndarray, kernel_size:int, nb_kernel:int, lr:float, activation, d_activation, bias:bool=True, mini:float=0, maxi:float=1):
        # classe héritée
        input_n = kernel_size**2
        output_n = nb_kernel
        super().__init__(
            input_n, output_n, 
            lr, activation, d_activation, bias, mini, maxi
        )
        self.kernel_size = kernel_size
        self.nb_kernel = nb_kernel # nombre total de filtre
        self.img_size = img_size
        self.output_size = max(0, img_size-kernel_size+1)


    def transform(self, imgs:np.ndarray):
        """
        Transforme une liste d'image d'entrée
        en input pour le perceptron  
        """
        n = self.output_size
        k = self.kernel_size      
        t_imgs = np.array([
            img[lig:lig+k, col:col+k].flatten()
            for img in imgs
            for lig in range(n)
            for col in range(n)
        ])
        return t_imgs 
    
    
    def transform_k(self, imgs:np.ndarray):
        """
        Transforme la liste d'image d'intermédiaire de calcul
        en entrée pour l'apprentissage
        """
        n = self.output_size
        nk = self.nb_kernel

        A = imgs
        A = A.reshape((-1, nk, n**2))
        A = A.transpose([0, 2, 1])
        A = A.reshape((-1, nk))
        return A


    def detransform_k(self, t_imgs:np.ndarray, nb_donnee:np.ndarray, nb_images:int):
        """
        Transforme la sortie
        en une liste d'image       
        """
        n = self.output_size
        nk = self.nb_kernel
        
        A = t_imgs
        A = A.reshape((nb_donnee*nb_images, n**2, nk))
        A = A.transpose([0, 2, 1])
        imgs = A.reshape((nb_donnee, nb_images*nk, n, n))
        return imgs



    def calculate(self, input_data:np.ndarray):
        # input est de taille (nb_donne, nb_images, hauteur, largeur)
        # On suppose (hauteur = largeur) : image carré
        """
        Calcule la sortie
        """
        # Ajout du biais
        nb_donnee = len(input_data)
        nb_images = len(input_data[0])
        input_data_k = np.concatenate([
            self.transform(input_data[i]) 
            for i in range(nb_donnee)
        ])

        if self.bias:
            self.input_data = np.concatenate(
                (input_data_k, np.ones((len(input_data_k), 1))), 
                axis=1
            ) 
        else:
            self.input_data = np.concatenate(
                (input_data_k, np.zeros((len(input_data_k), 1))), 
                axis=1
            ) 
        y1 = np.dot(self.input_data, self.weight)
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output = z1
        y1 = self.detransform_k(y1, nb_donnee, nb_images)
        z1 = self.detransform_k(z1, nb_donnee, nb_images)
        return (y1, z1)


    def learn(self, e_2:np.ndarray):
        """
        Permet de mettre à jour les poids weigth
        """
        shape = e_2.shape
        e_2 = self.transform_k(e_2)
        e1 = e_2 / (self.input_n+1) 
        e1 = e1 * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0


class Flatten:
    """
    Cette classe permet de faire le lien 
    entre les couches Convolutional 
    et les couches Layer
    """
    def __init__(self, img_size):
        self.img_size = img_size
        self.output_n = img_size**2
    

    def calculate(self, imgs):
        nb_donne = len(imgs)
        nb_img = len(imgs[0])

        flat = imgs.reshape((
            nb_donne, 
            nb_img*self.img_size*self.img_size
        ))
        return flat, flat
    

    def learn(self, e_2):
        nb_donne = len(e_2)
        e_0 = e_2.reshape((
            nb_donne, 
            -1, 
            self.img_size, self.img_size
        ))
        return e_0