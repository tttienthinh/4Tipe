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
    return 1 / (1 + np.exp(-np.clip(x, -5, 5)))

def d_sigmoid(x):
    """
    Dérivé de Sigmoid
    """
    return x * (1-x)

"""
Softmax
"""
def softmax(x):
    exp = np.exp(x) + 1e-5
    return exp / exp.sum(axis=-1, keepdims=True)

def d_softmax(x):
    s = softmax(x)
    return s * (1 - s)

"""
https://sgugger.github.io/a-simple-neural-net-in-numpy.html
https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
"""