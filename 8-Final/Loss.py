import numpy as np
import matplotlib.pyplot as plt

"""
Mean Square Error
"""
def mse(predicted_output, target_output):
    """
    Mean Square Error function
    """
    return ((predicted_output - target_output) ** 2).mean()

def d_mse(predicted_output, target_output):
    """
    Derivate Mean Square Error function
    """
    return predicted_output - target_output

"""
Cross-Entropy Loss
https://levelup.gitconnected.com/killer-combo-softmax-and-cross-entropy-5907442f60ba#aee4
https://gombru.github.io/2018/05/23/cross_entropy_loss#losses
https://www.tensorflow.org/api_docs/python/tf/keras/metrics/categorical_crossentropy
"""
def cross_entropy(predicted_output, target_output):
    return -(target_output * np.log(np.abs(predicted_output))).sum(axis=-1).mean()

def d_cross_entropy(predicted_output, target_output):
    return predicted_output - target_output
"""
https://youtu.be/xBEh66V9gZo?t=973
"""

"""
https://sgugger.github.io/a-simple-neural-net-in-numpy.html#cross-entropy-cost
"""