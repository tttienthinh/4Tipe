import numpy as np













# Mean Square Error
def mse(predicted_output, target_output):
    return ((predicted_output - target_output) ** 2).mean()

def d_mse(predicted_output, target_output):
    return predicted_output - target_output


# Cross-Entropy Loss
def cross_entropy(predicted_output, target_output):
    val_log = np.log(np.clip(np.abs(predicted_output), 1e-3, 1))
    return -(target_output * val_log).sum(axis=-1).mean()

def d_cross_entropy(predicted_output, target_output):
    return predicted_output - target_output

