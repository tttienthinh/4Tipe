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
    Au lieu de prendre la représentation de nb_kernel de taille kernel_size*kernel_size que l'on applique aux images.

    Je vais représenter les kernels sous la forme d'un Layer avec : 
        - input_n = kernel_size**2
        - output_n = nb_kernel
    """
    def __init__(self, img_size, kernel_size=3, nb_kernel=4, lr=0.1, activation=Activation.relu, d_activation=Activation.d_relu, biais=False, mini=0, maxi=1):
        input_n = kernel_size**2
        output_n = nb_kernel
        super().__init__(input_n, output_n, lr, activation, d_activation, biais, mini, maxi)

        self.kernel_size = kernel_size # matrice carrée de taille kernel_size*kernel_size
        self.nb_kernel = nb_kernel # nombre total de filtre
        self.img_size = img_size
        self.taille_sortie = max(0, img_size-kernel_size+1)

    def transform(self, imgs, taille_sortie=None, k=None):
        """
        Transforme une liste d'image d'entrée
        en input pour le perceptron  
        """
        if taille_sortie is None:
            taille_sortie = self.taille_sortie
            k = self.kernel_size      
        t_imgs = np.array([
            img[lig:lig+k, col:col+k].flatten()
            for img in imgs
            for lig in range(taille_sortie)
            for col in range(taille_sortie)
        ])
        return t_imgs 
        # doit être de taille 
        # (nb_imgs*taille_sortie*taillle_sortie), kernel_size**2
    
    def detransform(self, t_imgs, img_size=None, taille_sortie=None, k=None):
        """
        Detransforme une sortie de kernel
        en image
        """
        if img_size is None:
            img_size = self.img_size
            taille_sortie = self.taille_sortie
            k = self.kernel_size
        n = len(t_imgs)
        assert n % (taille_sortie**2)==0
        nb_imgs = int(n/(taille_sortie**2))
        imgs = [
            [
                [
                    []
                    for col in range(img_size)
                ]
                for lig in range(img_size)
            ]
            for i_img in range(nb_imgs)
        ]
        for i_img in range(nb_imgs):
            for lig_sortie in range(taille_sortie):
                for col_sortie in range(taille_sortie):
                    indice_flatten = (i_img*taille_sortie + lig_sortie)*taille_sortie + col_sortie
                    for lig_i in range(k):
                        for col_i in range(k):
                            indice_kernel = lig_i*k + col_i
                            lig = lig_sortie+lig_i
                            col = col_sortie+col_i

                            imgs[i_img][lig][col].append(
                                t_imgs[indice_flatten][indice_kernel]
                            )
        for i_img in range(nb_imgs):
            for lig in range(img_size):
                for col in range(img_size):
                    l = imgs[i_img][lig][col]
                    imgs[i_img][lig][col] = sum(l)/len(l)
        return np.array(imgs)
    
    def detransform_k(self, t_imgs, nb_donnee, nb_images, taille_sortie=None, nb_kernel=None):
        if taille_sortie is None:
            taille_sortie = self.taille_sortie
            nb_kernel = self.nb_kernel
        
        A = t_imgs
        A = A.reshape((nb_donnee*nb_images, taille_sortie**2, nb_kernel))
        A = A.transpose([0, 2, 1])
        imgs = A.reshape((nb_donnee, nb_images*nb_kernel, taille_sortie, taille_sortie))
        return imgs

    """ Enchainement
    (nb_donnee*nb_images, taille_sortie**2, nb_kernel)
    (nb_donnee*nb_images, nb_kernel, taille_sortie**2)
    (nb_donnee*nb_images*nb_kernel, taille_sortie, taille_sortie)
    """

    def transform_k(self, imgs, nb_donnee, nb_images, taille_sortie=None, nb_kernel=None):
        if taille_sortie is None:
            taille_sortie = self.taille_sortie
            nb_kernel = self.nb_kernel

        A = imgs
        A = A.reshape((-1, nb_kernel, taille_sortie**2))
        A = A.transpose([0, 2, 1])
        A = A.reshape((-1, nb_kernel))
        return A


    def calculate(self, input_data):
        # input est de taille (nb_donne, nb_images, hauteur, largeur)
        # On suppose (hauteur = largeur) : image carré
        """
        Calcule la sortie
        """
        # Ajout du biais
        nb_donnee = len(input_data)
        nb_images = len(input_data[0])
        input_data_k = np.concatenate([
            self.transform(
                input_data[i], 
                self.taille_sortie,
                self.kernel_size
            ) 
            for i in range(nb_donnee)
        ])

        if self.biais:
            self.input_data = np.concatenate((input_data_k, np.ones((len(input_data_k), 1))), axis=1) 
        else:
            self.input_data = np.concatenate((input_data_k, np.zeros((len(input_data_k), 1))), axis=1) 
        y1 = np.dot(self.input_data, self.weight)
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output = z1
        y1 = self.detransform_k(y1, nb_donnee, nb_images, self.taille_sortie, self.nb_kernel)
        z1 = self.detransform_k(z1, nb_donnee, nb_images, self.taille_sortie, self.nb_kernel)
        return (y1, z1)


    def learn(self, e_2):
        """
        Permet de mettre à jour les poids weigth
        """
        shape = e_2.shape
        e_2 = self.transform_k(
            e_2, shape[0], shape[1], self.taille_sortie, self.nb_kernel
        )
        e1 = e_2 / (self.input_n+1) * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0

class Flatten:
    def __init__(self, img_size):
        self.img_size = img_size
        self.output_n = img_size**2
    
    def calculate(self, imgs):
        nb_donne = len(imgs)
        nb_img = len(imgs[0])

        flat = imgs.reshape((nb_donne, nb_img*self.img_size*self.img_size))
        return flat, flat
    
    def learn(self, e_2):
        nb_donne = len(e_2)
        e_0 = e_2.reshape((nb_donne, -1, self.img_size, self.img_size))
        return e_0
        


if __name__ == "__main__":
    # Traitement des données
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    y_train = np.zeros((len(Y_train), 10))
    y_train[np.arange(len(Y_train)), Y_train] = 1 # to categorical
    y_test = np.zeros((len(Y_test), 10))
    y_test[np.arange(len(Y_test)), Y_test] = 1 # to categorical 

    # cela permet de transformer la sortie en une liste [0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0] 
    # avec un 1 à l'indice n
    # par exemple si le nombre cherché est 2 : [0, 0, 1, 0, 0, 0, 0, 0 ,0, 0, 0] 

    x_train = X_train.reshape(-1, 1, 28, 28)/255 # 28*28 = 784
    x_test = X_test.reshape(-1, 1, 28, 28)/255

    """
    Convolutional(img_size=26, kernel_size=3, nb_kernel=4), # 28-kernel_size+1=26, nb_image=3
    Flatten(24), # 26-3+1=24, nb_image=9            
    """
    # Creation du model
    model = ModelClassification([
            # LayerOptimizer(784, 256, lr=0.5, gamma=0.5, activation=sigmoid, d_activation=d_sigmoid),
            Convolutional(img_size=28, kernel_size=3, nb_kernel=4),
            Flatten(26), # 28-kernel_size+1=26, nb_image=3
            LayerOptimizer(2704, 10, lr=0.5, gamma=0.5, activation=softmax, d_activation=d_softmax),
        ],
        loss_function=cross_entropy,
        d_loss_function=d_cross_entropy
    )


    # Entrainement
    losses = []
    accs = []
    epochs = 100
    for epoch in range(epochs):
        y, loss, acc = model.backpropagation(x_train[:100], y_train[:100])
        losses.append(loss)
        accs.append(acc*100)
        if epoch%5 == 0:
            print(f"Epoch {epoch} : {round(acc*100, 2)}% Accuracy")

    # Affichage résultat
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].plot(losses)
    axs[0].set_title("Courbe d'erreur")
    axs[1].plot(accs)
    axs[1].set_title("Taux de précision (%)")
    axs[1].set_ylim([0, 100])
    for i in range(2):
        axs[i].grid()
    plt.savefig("Accs.jpg", dpi=400)
    """

    print(model.backpropagation(x_train, y_train)[1:])
    print(model.backpropagation(x_test, y_test)[1:])
    model.backpropagation(x_test, y_test)[0].argmax(axis=-1)
    """


    
    
    fig = plt.figure(figsize=(15,10))
    start = 40
    end = start + 40
    test_preds = model.backpropagation(x_test[start:end], y_test[start:end])[0].argmax(axis=-1)
    for i in range(40):  
        ax = fig.add_subplot(5, 8, (i+1))
        ax.imshow(X_test[start+i], cmap=plt.get_cmap('gray'))
        if Y_test[start+i] != test_preds[i]:
            ax.set_title('Prediction: {res}'.format(res=test_preds[i]), color="red")
        else:
            ax.set_title('Prediction: {res}'.format(res=test_preds[i]))
        plt.axis('off')
    plt.savefig("Resultat.jpg", dpi=400)

    
