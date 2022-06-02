import numpy as np
import matplotlib.pyplot as plt

# Nous allons ici créer un Neuronne Linéaire sans fonction d'activation, et nous allons l'entrainer à multiplier par la valeur cible.










class NeuroneLineaire:
    def __init__(self, lr=0.8, cible=0.5):
        self.w = np.random.random()
        self.lr = lr # le learning rate
        self.liste_w = [] # l'historique des poids
        self.liste_e = [] # l'historique des erreurs
        self.cible = cible # le poids ciblé par l'entrainement
        

    def calcul(self, x):
        return self.w * x # prédiction du résultat
    

    @staticmethod
    def erreur(y_, y): # sortie calculé, sortie souhaitée
        return (y - y_) ** 2
    

    @staticmethod
    def d_erreur(x, y_, y): # sortie calculé, sortie souhaitée
        return (2*(y_-y) * 1 * x).mean()
    

    def validation(self):
        e = self.erreur(self.w, self.cible)
        self.liste_w.append(self.w)
        self.liste_e.append(e)
        

    def calcul_dw(self, x, y):
        if type(x) == int:
            x, y = np.array([x]), np.array([y])
        y_ = self.calcul(x)
        dw = self.d_erreur(x, y_, y)
        return dw
    

    def retropropagation(self, x, y):
        dw = self.calcul_dw(x, y)
        self.w -= self.lr * dw # Mise à jour du poids
    

    def plot(self, ax, title, mini, maxi):
        cible = self.cible
        x = np.linspace(min(mini, min(self.liste_w)), 
                        max(maxi, max(self.liste_w)), 
                        1_000)
        y = self.erreur(x, self.cible)
        ax.plot(x, y, label="Courbe d'erreur")
        ax.plot(
            self.liste_w, self.liste_e, 'o-', lw=3, 
            label="Apprentissage"
        )
        ax.set_title(title)
        ax.grid()
        ax.legend()
