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

