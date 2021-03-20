"""
https://www.youtube.com/watch?v=wQ8BIBpya2k
"""
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(tf.keras.utils.normalize(X_test, axis=1))