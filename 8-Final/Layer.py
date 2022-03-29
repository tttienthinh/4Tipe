import numpy as np
import Activation
"""
17/03/2021 tranthuongtienthinh
Feed Forward model pour résoudre le xor
algorithme POO
"""


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


class LayerOptimizer(Layer):
    """
    On hérite de la class layer, car toutes les fonctions sont les mêmes
    Sauf l'apprentissage qui invoque un taux d'apprentissage variable
        Pour cela on utilise la variable gamma
    """

    def __init__(self, input_n=2, output_n=2, lr=0.1, activation=Activation.sigmoid, d_activation=Activation.d_sigmoid, gamma=0.5, biais=True, mini=0, maxi=1):
        super().__init__(input_n, output_n, lr, activation, d_activation, biais, mini, maxi)
        self.gamma = gamma
        self.dw_moment = np.zeros((input_n+1, output_n))

    def learn(self, e_2):
        """
        Permet de mettre à jour les poids weigth
        """
        e1 = e_2 / (self.input_n+1) * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        """
        La différence se trouve ici
        """
        self.dw_moment =  self.gamma * self.dw_moment + dw1.T * self.lr
        self.weight -= self.dw_moment
        return e_0