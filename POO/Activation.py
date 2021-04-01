import numpy as np
import matplotlib.pyplot as plt

"""
Linéaire
"""
def lineaire(x):
    """
    Fonction activation Linéaire
    """
    return x

def d_lineaire(x):
    """
    Dérivé Linéaire
    """
    return np.ones(x.shape)


"""
Sigmoid
"""
def sigmoid(x):
    """
    Fonction Sigmoid
    """
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    """
    Dérivé de Sigmoid
    """
    return x * (1-x)

