import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer

"""
17/03/2021 tranthuongtienthinh
Feed Forward model pour résoudre le xor
algorithme POO
"""


class Model:

    def __init__(self, layers=[]):
        self.layers = layers
        self.loss = []
        self.lr = 0.1

    def __str__(self):
        return str([layer.w1 for layer in self.layers])

    def shape(self):
        return [layer.shape() for layer in self.layers]

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        z_ = x  # y_ is predicted data
        for layer in self.layers:
            y_, z_ = layer.calculate(z_)
        return z_

    def predict_loss(self, x, y):  # y is expected data
        y_ = self.predict(x)  # y_ is predicted data
        loss = ((y_ - y) ** 2).mean()
        return y_, loss

    def backpropagation(self, x, y):
        y_, loss = self.predict_loss(x, y)
        d_loss = (y_ - y)  # dérivé de loss dy_/dy
        # Entrainement des layers
        for i in range(len(self.layers)):
            d_loss = self.layers[-i - 1].learn(d_loss)
        self.loss.append(loss)
        return y_, loss


if __name__ == "__main__":
    # training data XOR !!!
    train_input = np.array([
        [0, 0], [1, 1],
        [0, 1], [1, 0]
    ])
    train_output = np.array([0, 0, 1, 1]).reshape((4, 1))

    # model testing
    l1 = Layer(2, 2)
    l2 = Layer(2, 1)
    """
    np.random.seed(2)
    l1.w1 = np.random.rand(2, 2)
    l2.w1 = np.random.rand(2, 1)
    """
    model = Model([
        l1,
        l2,
    ])
    losses = []
    for i in range(100_000):
        y, loss = model.backpropagation(train_input, train_output)
        losses.append(loss)
    plt.plot(losses)
    plt.show()
    print(model.predict(train_input))

