1)
  \item[n°15] malaise, hémorragie, brûlure, intoxication
  \item[n°17] violences, agression, vol, cambriolage
  \item[n°18] incendie, gaz, effondrement, électrocution
  
Par l'apprentissage du réseau de neurones, j'ai trouvé que certains mots étaient mieux appris que d'autre, notamment le mot "INCENDIE". 
Je pense que cela est dû au fait que sa prononciation ne diffère pas grandement d'une personne à une autre.
Est-il vrai de dire que certain mot comme "INCENDIE" sont moins sujet à des différence de prononciation contrairement par exemple à "AGRESSION" qui peut se prononcer "agréssion", "agrèssion", "agressïon"...
Si oui, cela porte-t-il un nom ? Et comment reperer ce genre de mot ?

1) oui t'as raison ça s'appelle la variation de la parole. En linguistique on fait une différence entre 'langue' et 'parole' : la langue c'est si tu veux la théorie, c'est le code commun qu'on a tous dans une communauté. La parole c'est ce qu'on produit. Elle est différente puisqu'elle est hétérogène et va alors varier en fonction de plein de paramètres, qu'ils soient géographiques (accents), mais aussi d'autres raison comme l'économie : on essaie toujours d'aller au plus vite et avec le moins d'efforts possibles, donc on va enlever des sons, combiner d'autres. Mais le cerveau fonctionne sur un mode de compréhension globale et pas son par son. Ce qui permet de deviner. 

Il y a des situation où on modifie le son comme par exemple le mot "Afghanistan" qui est prononcé toujours "Avganistan" par assimilation (combinaison) du f et g = ça donne un v et g

Tout ça pour dire que dans le mot "incendie" il y a peu de chance de variation de parole car les syllabes ne s'assimilent pas entre elles. Seule variation que tu peux avoir c'est le "in" qui peut parfois se prononcer [un] ou [in] mais rare (en fonction des génération beaucoup)


2)
Est ce que je peux faire des rectangles autour des taches de vives couleurs pour dire que ce sont les formants ou est ce faux, ce ne sont que les syllabes.
L'image en question se trouve à ce lien :
https://github.com/tttienthinh/4Tipe/raw/main/Presentation/Beamer/Affiche/0-ReconnaissanceVocale/1-Incendie-3.jpg

Mon système est un réseau de neurones, c'est une boite noire : 
Spectrogramme -> boite noire -> mot prononcé
Je ne peux pas avoir accès aux étapes intermédiaires de la reconnaissance vocale, mais étant donné comment je l'adapte à mon problèmes, je me dis que forcement il doit décomposer le spectrogramme qu'il a entré pour ensuite recomposer le mot.

[d] est une consonne occlusive donc c'est une fine barre

En fait ce que tu vois en basse fréquence c'est du voisement = vibration des cordes vocales. Toutes les voyelles sont voisées par définition. 
Toutes les consonnes ne le sont pas. 
Par exemple [d] est voisé alors que [t] non, [b] est voisé alors que [p] non,... 
La seule différence entre ces sons sera alors la présence de cette barre de voisement

3)
J'utilise le tranfert d'apprentissage, la dernière couche d'un réseau de neurones entrainé est retirée pour la remplacer et l'adapter à mon problème à moi : ça fonctionne bien, voici un schéma
https://github.com/tttienthinh/4Tipe/raw/main/Presentation/Beamer/Affiche/6-Mot/2-transfert.png
J'ai envie de dire que la première partie du réseau de neurones permet de repérer les formants, puis de reconnaitre lequel c'est. Finalement la dernière couche permet de réunir toutes les informations pour donner en sortie le mot prononcé.
