import numpy as np
import matplotlib.pyplot as plt

"""
Linéaire
https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4
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
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def d_sigmoid(x):
    """
    Dérivé de Sigmoid
    """
    # f_x = sigmoid(x)
    f_x = x
    return f_x * (1-f_x)

"""
Tanh
"""
def tanh(x):
    """
    Fonction Tanh
    """
    return np.tanh(x)

def d_tanh(x):
    """
    Dérivé de Tanh
    """
    return 1-tanh(x)**2

"""
ReLU
"""
def relu(x):
    """
    Fonction ReLU
    """
    return np.clip(x, 0, np.inf)

def d_relu(x):
    """
    Dérivé de ReLU
    """
    return 1*(x>0)

"""
Softmax
"""
def softmax(x):
    exp = np.exp(x-x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def d_softmax(x):
    # s = softmax(x)
    # return s * (1 - s)
    return np.ones(x.shape)

"""
https://sgugger.github.io/a-simple-neural-net-in-numpy.html
https://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
"""

if __name__ == "__main__":
    """
    Affichage
    """
    save_path = "/home/tttienthinh/Documents/Programmation/4Tipe/8-Final/Presentation/Beamer/Affiche/2-Activation-Gradient/"

    x = np.linspace(-5, 5, 1001)
    y = sigmoid(x)
    d_y = d_sigmoid(x)
    plt.plot(x, y, label="Sigmoïde")
    plt.plot(x, d_y, label="dérivée Sigmoïde")
    plt.axis((-5,5,-1.1,1.1))
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}Sigmoide.png")
    plt.clf()

    x = np.linspace(-5, 5, 1001)
    y = tanh(x)
    d_y = d_tanh(x)
    plt.plot(x, y, label="Tanh")
    plt.plot(x, d_y, label="dérivée Tanh")
    plt.axis((-5,5,-1.1,1.1))
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}Tanh.png")
    plt.clf()

    x = np.linspace(-5, 5, 1001)
    y = relu(x)
    d_y = d_relu(x)
    plt.plot(x, y, label="ReLU")
    plt.plot(x, d_y, label="dérivée ReLU")
    plt.axis((-5,5,-1.1,1.1))
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}ReLU.png")
    plt.clf()