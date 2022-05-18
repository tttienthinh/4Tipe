import numpy as np
import Activation


class Layer:
    def __init__(self, input_n=2, output_n=2, lr=0.1, activation=Activation.sigmoid, d_activation=Activation.d_sigmoid, biais=True, mini=0, maxi=1):
        """
        Crée un layer de n neuronne connecté aux layer de input neuronnes
        """
        # input_n le nombre d'entrée du neuronne
        # output_n le nombre de neuronne de sortie
        self.weight = np.random.rand(input_n+1, output_n)*(maxi-mini)+mini
        self.biais = biais
        self.input_n = input_n
        self.output_n = output_n
        self.lr = lr # learning rate

        # the name of the layer is 1
        # next one is 2 and previous 0
        self.predicted_output_ = 0
        self.predicted_output  = 0
        self.input_data = 0

        # Fonction d'activation
        self.activation = activation
        self.d_activation = d_activation

    def next(self):
        return self.output_n

    def calculate(self, input_data):
        """
        Calcule la sortie
        """
        # Ajout du biais
        if self.biais:
            self.input_data = np.concatenate((input_data, np.ones((len(input_data), 1))), axis=1) 
        else:
            self.input_data = np.concatenate((input_data, np.zeros((len(input_data), 1))), axis=1) 
        y1 = np.dot(self.input_data, self.weight)
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output = z1
        return y1, z1

    def learn(self, e_2):
        """
        Permet de mettre à jour les poids weigth
        """
        e1 = e_2 / (self.input_n+1) * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0



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
        return (
            self.detransform_k(y1, nb_donnee, nb_images, self.taille_sortie, self.nb_kernel),
            self.detransform_k(z1, nb_donnee, nb_images, self.taille_sortie, self.nb_kernel)
        )


    def learn(self, e_2):
        """
        Permet de mettre à jour les poids weigth
        """
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

    def next(self):
        return self.output_n
    
    def calculate(self, imgs):
        nb_donne = len(imgs)
        nb_img = len(imgs[0])

        flat = imgs.reshape((nb_donne, nb_img*self.img_size*self.img_size))
        return flat
    
    def learn(self, e_2):
        nb_donne = len(e_2)
        e_0 = e_2.reshape((nb_donne, -1, self.img_size, self.img_size))
        return e_0
        


if __name__ == "__main__":
    conv = Convolutional(img_size=5, kernel_size=3, nb_kernel=4, lr=0.1, activation=Activation.relu, d_activation=Activation.d_relu, biais=False, mini=0, maxi=1)
    """
    imgs = np.ceil(np.random.rand(3, 5, 5)*100)
    imgs
    t_imgs = conv.transform(imgs)
    imgs_ = conv.detransform(t_imgs)
    imgs_
    """

    data = np.ceil(np.random.rand(10, 3, 5, 5)*100)
    data
    conv.calculate(data)    


    layer = Layer(input_n=2, output_n=2)
    data_l = np.ceil(np.random.rand(10, 2)*100)
    layer.calculate(data_l)

    """
    A = np.ceil(np.random.rand(24, 5)*100)
    B = A.reshape((6, 4, 5))
    C = B.transpose([0, 2, 1])
    C.reshape((2, 15, 2, 2))



    C = np.ceil(np.random.rand(2, 3)*100)
    np.concatenate([A, B])    
    """