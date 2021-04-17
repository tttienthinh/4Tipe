import numpy as np
import numpy.random as rd
from matplotlib import pyplot as plt

figsize=(5, 5)
sigmo = lambda x: 1 / (1 + np.exp(-x))
sigmo_prime = lambda x: np.exp(-x) / ((1 + np.exp(-x)) ** 2)

def coeff_alea():
    return 2 * rd.random() - 1


def creer_reseau(dimensions):
    
    """Entrées: dimension est une liste d'entiers. Sa longueur donne le nombre de couches.
    Pour tout i, dimensions[i] donne le nombre de neurones par couche.
    Sortie: Une liste de listes de même longueur que dimensions. Pour tout i, j, reseau[i][j] est une liste de
    flottants donnant les coefficients du j-ème neurone de la i-ème couche. Ce nombre de coefficients est égal à
    k = dimensions[i-1] + 1. Le dernier coefficient de reseau[i][j], soit reseau[i][j][k], vaut 1.
    """
    
    nb_couches = len(dimensions)
    reseau = [[[coeff_alea(), coeff_alea(), coeff_alea()] for i in range(0, dimensions[0])]]
    for i in range(1, nb_couches):
        reseau.append([[coeff_alea() for k in range(0, dimensions[i - 1])] + [coeff_alea()] for j in range(0, dimensions[i])])
    return reseau


def evaluation(coefficients, variables):
    n = len(coefficients)
    res = 0
    for i in range(0, n):
        res += coefficients[i] * variables[i]
    return res


def heavyside(x):
    if x > 0: 
        res = 1
    else:
        res = 0
    return res


def prevision(point, reseau):
    
    nb_couches = len(reseau)
    X = [[point[0], point[1], 1]] + [[0 for i in range(0, len(reseau[j]))] + [1] for j in range(0, nb_couches - 1)]
    X.append([0 for i in range(0, len(reseau[nb_couches - 1]))])
    
    for i in range(0, nb_couches):
        for j in range(0, len(reseau[i])):
            X[i + 1][j] = sigmo(evaluation(reseau[i][j], X[i]))
    
    return X


def erreur(echantillon, reseau):
    
    nb_couches = len(reseau)
    X = prevision(echantillon[0], reseau)
    liste_erreurs = [[0 for j in range(0, len(reseau[i]))] for i in range(0, nb_couches)]
    
    liste_erreurs[nb_couches - 1][0] = echantillon[1] - X[nb_couches][0]
    for i in range(2, nb_couches + 1):
        for j in range(0, len(reseau[nb_couches - i])):
            coeffs_erreurs = [reseau[nb_couches - i + 1][k][j] for k in range(0, len(reseau[nb_couches - i + 1]))]
            a = sigmo_prime(evaluation(reseau[nb_couches - i][j], X[nb_couches - i]))
            b = evaluation(coeffs_erreurs, liste_erreurs[nb_couches - i + 1])
            liste_erreurs[nb_couches - i][j] = a * b
    
    return liste_erreurs


def mise_a_jour(echantillon, reseau, X, t): # t est le learning rate
    nb_couches = len(reseau)
    liste_erreurs = erreur(echantillon, reseau)
    
    for i in range(0, nb_couches):
        for j in range(0, len(reseau[i])):
            for k in range(0, len(reseau[i][j])):
                reseau[i][j][k] += t * X[i][k] * liste_erreurs[i][j]
    return(reseau)


def apprentissage(donnees, reseau, nb_iterations, t):
    for i in range(0, nb_iterations):
        echantillon = donnees[rd.randint(0, len(donnees) - 1)]
        X = prevision(echantillon[0], reseau)
        mise_a_jour(echantillon, reseau, X, t)
    return reseau
      
    
def production_donnees_cercle(centre, rayon, nb_points, show=True):
    
    """Entrées: un couple de flottants (centre), un flottant strictement positif (rayon) et un entier et nb_points 
       Sortie: Une liste de nb_points listes de la forme [(x, y), categorie], où (x, y) est 
       un point du plan et categorie est un entier.
    """
    
    #Production des donnees
    donnees = []
    
    for i in range(0, nb_points):
        x = rd.random() * 20 - 10
        y = rd.random() * 20 - 10
        categorie = sigmo(rayon ** 2 - (x - centre[0]) ** 2 - (y - centre[1]) ** 2)
        donnees.append([(x, y), categorie])
    
    #Representation des donnees
    theta = np.linspace(0, 2 * np.pi, 1000)
    X = centre[0] * np.ones(1000) + rayon * np.cos(theta)
    Y = centre[1] * np.ones(1000) + rayon * np.sin(theta)
    
    categorie_1 = np.array([point[0] for point in donnees if point[1] > 0.5])
    categorie_0 = np.array([point[0] for point in donnees if point[1] < 0.5])
    if show:
        plt.figure(figsize=figsize)
        plt.plot(X, Y)
        if len(categorie_1) > 0:
            plt.plot(categorie_1[:, 0], categorie_1[:, 1], 'ro', color='blue')
        if len(categorie_0) > 0:
            plt.plot(categorie_0[:, 0], categorie_0[:, 1], 'ro', color='red')
        plt.show()
    return donnees


def representation_apprentissage_cercle(donnees, reseau, centre, rayon):
    
    """Entrées: un couple de flottants (centre), un flottant strictement positif (rayon) et un entier et nb_points 
       Sortie: Une liste de nb_points listes de la forme [(x, y), categorie], où (x, y) est 
       un point du plan et categorie est un entier.
    """
    previsions = [echantillon[:] for echantillon in donnees]
    for echantillon in previsions:
        echantillon[1] = prevision(echantillon[0], reseau).pop()[0]
    categorie_1 = np.array([point[0] for point in previsions if point[1] > 0.5])
    categorie_0 = np.array([point[0] for point in previsions if point[1] < 0.5])
    
    plt.figure(figsize=figsize)
    if len(categorie_1) > 0:
        plt.plot(categorie_1[:, 0], categorie_1[:, 1], 'ro', color='blue')
    if len(categorie_0) > 0:
        plt.plot(categorie_0[:, 0], categorie_0[:, 1], 'ro', color='red')
    
    theta = np.linspace(0, 2 * np.pi, 1000)
    X = centre[0] * np.ones(1000) + rayon * np.cos(theta)
    Y = centre[1] * np.ones(1000) + rayon * np.sin(theta)
    plt.plot(X, Y)
    plt.show()
    

def production_donnees_2d_2ineq(w1, w2, n):
    
    """Entrées: deux tableaux w1 et w2 contenant chacun trois flottants et un entier n
       Sortie: une liste dont les éléments sont des listes à deux éléments: le premier élément est un couple, 
       le second un entier.
       La fonction produit aléatoirement n couples identifiés à des points du plan et les classe en deux catégories:
           ceux qui vérifient w1[0] * x + w1[1] * y + w1[2] > 0 et w2[0] * x + w2[1] * x + w2[2] > 0 
           sont dans la catégorie 1, les autres dans la catégorie 0."""
           
    res = []
    for i in range(0, n):
        x = rd.random() * 20 - 10
        y = rd.random() * 20 - 10
        if w1[0] * x + w1[1] * y + w1[2] > 0 and w2[0] * x + w2[1] * y + w2[2] > 0:
            categorie = 1
        else:
            categorie = 0
        res.append([(x, y), categorie])
    return res


def representation_apprentissage_2ineq(donnees, w1, w2, reseau):
    
    previsions = [echantillon[:] for echantillon in donnees]
    for echantillon in previsions:
        echantillon[1] = prevision(echantillon[0], reseau).pop()[0]
    categorie_1 = np.array([point[0] for point in previsions if point[1] == 1])
    categorie_0 = np.array([point[0] for point in previsions if point[1] == 0])
    f = lambda x: -(w1[0] /w1[1]) * x - (w1[2] / w1[1])
    g = lambda x: -(w2[0] / w2[1]) * x - (w2[2] / w2[1])
    X = np.linspace(-10, 10, 1000)
    Y = f(X)
    Z = g(X)
    plt.plot(X, Y, X, Z)
    if len(categorie_1) > 0:
        plt.plot(categorie_1[:, 0], categorie_1[:, 1], 'ro', color='blue')
    if len(categorie_0) > 0:
        plt.plot(categorie_0[:, 0], categorie_0[:, 1], 'ro', color='red')
    plt.show()


if __name__ == "__main__":
    centre = (0, 0)
    rayon = 3
    donnees = production_donnees_cercle(centre, rayon, 10_000, False)

    # entrainement avec descente lr
    n1_reseau = creer_reseau([3, 3, 1])
    for epoch, lr in [(2_048, 0.6), (1_024, 0.5), (548, 0.3), (256, 0.2)]:
        n1_reseau = apprentissage(donnees, n1_reseau, epoch, lr)
    # entrainement normal
    n2_reseau = creer_reseau([3, 3, 1])
    n2_reseau = apprentissage(donnees, n2_reseau, (2_048 + 1_024 + 548 + 256), 0.6)

    # affichage
    donnees = production_donnees_cercle(centre, rayon, 1_000, False)
    representation_apprentissage_cercle(donnees, n1_reseau, centre, rayon)
    representation_apprentissage_cercle(donnees, n2_reseau, centre, rayon)