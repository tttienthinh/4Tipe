import numpy as np

"""
17/03/2021 tranthuongtienthinh
La class layer représente une couche de perceptron

"""







class Layer:
    def __init__(self, input_n:int, output_n:int, lr:float, activation, d_activation, bias:bool=True, mini:float=0, maxi:float=1):
        """
        Crée un layer de neurones
        """
        # input_n le nombre d'entrée du neurones
        # output_n le nombre de neurone de sortie
        self.weight = np.random.rand(
            input_n+1, 
            output_n
        )*(maxi-mini)+mini
        self.bias = bias
        self.input_n = input_n
        self.output_n = output_n
        self.lr = lr # learning rate (taux d'apprentissage)

        self.predicted_output_ = 0 # sortie avant activation
        self.predicted_output  = 0 # sortie 
        self.input_data = 0

        # Fonction d'activation
        self.activation = activation
        self.d_activation = d_activation


    def calculate(self, input_data:np.ndarray):
        """
        Calcule la sortie
        """
        # Ajout du biais
        if self.bias:
            self.input_data = np.concatenate(
                (input_data, np.ones((len(input_data), 1))), 
                axis=1
            ) 
        else:
            self.input_data = np.concatenate(
                (input_data, np.zeros((len(input_data), 1))), 
                axis=1
            ) 
        y1 = self.input_data@self.weight
        z1 = self.activation(y1)
        self.predicted_output_ = y1
        self.predicted_output  = z1
        return y1, z1


    def learn(self, e_2:np.ndarray):
        """
        Permet de mettre à jour les poids "weight"
        """
        e1 = e_2 / (self.input_n+1) 
        e1 = e1 * self.d_activation(self.predicted_output)
        # e_0 est gardé pour entrainer la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        self.weight -= dw1.T * self.lr
        return e_0


class LayerOptimizer(Layer):
    """
    On hérite de la class Layer, 
        car toutes les fonctions sont les mêmes
    Sauf l'apprentissage 
        qui invoque un taux d'apprentissage variable
        on utilise la variable gamma
    """

    def __init__(self, input_n:int, output_n:int, lr:float, activation, d_activation, bias:bool=True, mini:float=0, maxi:float=1, gamma:float=0.5):
        # classe héritée
        super().__init__(
            input_n, output_n, 
            lr, activation, d_activation, bias, mini, maxi
        )
        self.gamma = gamma
        self.dw_moment = np.zeros((input_n+1, output_n))


    def learn(self, e_2:np.ndarray):
        """
        Permet de mettre à jour les poids weigth 
        en prenant en compte le momentum
        """
        e1 = e_2 / (self.input_n+1) 
        e1 = e1 * self.d_activation(self.predicted_output)
        # e_0 est pour l'entrainement de la couche précédente
        e_0 = np.dot(e1, self.weight.T)[:, :-1]
        dw1 = np.dot(e1.T, self.input_data)
        
        # La différence est ci-dessous
        self.dw_moment  = self.gamma * self.dw_moment
        self.dw_moment += dw1.T * self.lr
        self.weight -= self.dw_moment
        return e_0
