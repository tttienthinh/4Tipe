import numpy as np
import matplotlib.pyplot as plt
"""
17/03/2021 tranthuongtienthinh
Feed Forward model pour résoudre le xor
algorithme POO
"""


class Layer:
    def __init__(self, input_n=2, output_n=2, lr=0.1):
        """
        Crée un layer de n neuronne connecté aux layer de input neuronnes
        """
        self.w1 = np.random.rand(input_n, output_n)
        self.input_n = input_n
        self.output_n = output_n
        self.lr = lr
        # the name of the layer is 1
        # next one is 2 and previous 0
        self.y1 = 0
        self.z1 = 0
        self.z0 = 0

    def __str__(self):
        return str(self.w1)

    def calculate(self, input_data):
        """
        Calcule la sortie
        """
        self.z0 = input_data
        y1 = np.dot(input_data, self.w1)
        z1 = self.activation(y1)
        self.y1 = y1
        self.z1 = z1
        return y1, z1

    def learn(self, e_2):
        """
        Permet de mettre à jour les weigths
        """
        e1 = e_2 * self.d_activation(self.z1)
        # e_0 is for the next layer
        e_0 = np.dot(e1, self.w1.T)
        dw1 = np.dot(e1.T, self.z0)
        self.w1 -= dw1.T * self.lr
        return e_0

    @staticmethod
    def activation(x):
        """
        Fonction d'activation choisi avec sigmoid par défaut mais modifiable
        """
        return 1/(1 + np.exp(-x))

    @staticmethod
    def d_activation(x):
        """
        Dérivé de sigmoid
        """
        return x * (1-x)

