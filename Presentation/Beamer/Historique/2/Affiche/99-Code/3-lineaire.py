













# Petit learning rate
reseau = NeuroneLineaire(lr=0.1, cible=0.75)
reseau.w = 0.1
while abs(2*(0.75-reseau.calcul(1))) > 0.1:
    reseau.validation()
    reseau.retropropagation(1, 0.75)
reseau.validation()
reseau.plot(
    ax=plt, title=f"Petit taux : t = 0.1", mini=0, maxi=1
)