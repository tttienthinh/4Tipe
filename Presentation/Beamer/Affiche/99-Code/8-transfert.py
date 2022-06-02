import tensorflow as tf

"""
Transfert d'apprentissage
Vous avez besoin du fichier model.h5 pour exécuter
"""








import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
model.trainable = False
model_adapt = tf.keras.Sequential(
    [tf.keras.layers.Input(shape=(374, 129, 1))] 
    + model.layers[0:-1] 
    + [tf.keras.layers.Dense(12, activation="softmax", 
        name="adaptation")]
)
print(model_adapt.summary())
